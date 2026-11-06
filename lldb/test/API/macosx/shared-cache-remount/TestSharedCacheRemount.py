"""
Test that lldb stops using an inferior's shared cache once the volume that
held it is gone, even when a cache with the same UUID comes back at a new
path. A long-lived lldb can see this when a simulator runtime is unmounted
and remounted between debug sessions.
"""

import ctypes
import json
import os
import plistlib
import shutil
import subprocess
import time
import uuid

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

# The DriverKit shared cache is small and its UUID differs from the shared
# cache lldb itself runs from, so lldb has to map it from the file.
CACHE_DIRS = [
    "/System/Cryptexes/OS/System/DriverKit/System/Library/dyld",
    "/System/DriverKit/System/Library/dyld",
]
CACHE_NAME = "dyld_shared_cache_arm64e"
DYLIB = "/System/DriverKit/usr/lib/system/libdyld.dylib"
MH_MAGIC_64 = 0xFEEDFACF


def whole_disk(entities):
    """Returns the device of a disk image (/dev/diskN), given the
    system-entities hdiutil reports for it."""
    devices = [e["dev-entry"] for e in entities if "dev-entry" in e]
    return min(devices, key=lambda d: (len(d), d)) if devices else None


def attach(image, mount_point):
    """Attaches a disk image at mount_point, and returns its device."""
    # hdiutil warns on stderr that these options are deprecated; only show its
    # stderr if the attach fails.
    result = subprocess.run(
        ["hdiutil", "attach", "-plist", "-nobrowse", "-mountpoint", mount_point]
        + [image],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "hdiutil attach failed: %s" % result.stderr.decode(errors="replace")
        )
    return whole_disk(plistlib.loads(result.stdout).get("system-entities", []))


def attached_images_under(directory):
    """Returns the devices of the attached disk images whose files are under
    directory."""
    info = plistlib.loads(subprocess.check_output(["hdiutil", "info", "-plist"]))
    devices = []
    for image in info.get("images", []):
        if image.get("image-path", "").startswith(directory):
            devices.append(whole_disk(image.get("system-entities", [])))
    return [d for d in devices if d]


def force_detach(device, attempts=3):
    """Force-detaches the disk image at device, retrying as simdiskimaged
    does, since a detach can be refused while the volume is briefly busy.
    Returns whether it worked."""
    for attempt in range(attempts):
        if attempt:
            time.sleep(2)
        if subprocess.call(["hdiutil", "detach", "-force", "-quiet", device]) == 0:
            return True
    return False


class Responder(MockGDBServerResponder):
    def __init__(self, cache_path, cache_uuid):
        MockGDBServerResponder.__init__(self)
        self.cache_path = cache_path
        self.cache_uuid = cache_uuid

    def respond(self, packet):
        if packet == "qProcessInfo":
            return "pid:a860;parent-pid:d2a0;real-uid:1f5;real-gid:14;effective-uid:1f5;effective-gid:14;cputype:100000c;cpusubtype:2;ptrsize:8;ostype:macosx;vendor:apple;endian:little;"
        if packet == "jGetSharedCacheInfo:":
            return "OK"
        if packet.startswith("jGetSharedCacheInfo:"):
            return escape_binary(
                json.dumps(
                    {
                        "shared_cache_base_address": 0x180000000,
                        "shared_cache_uuid": self.cache_uuid,
                        "no_shared_cache": False,
                        "shared_cache_private_cache": False,
                        "shared_cache_path": self.cache_path,
                    }
                )
            )
        return MockGDBServerResponder.respond(self, packet)

    def qHostInfo(self):
        # Without an os_version, lldb uses the old dynamic loader plugin, which
        # doesn't ask for jGetLoadedDynamicLibrariesInfos or the shared cache.
        return "cputype:16777228;cpusubtype:2;ostype:macosx;vendor:apple;os_version:27.0;endian:little;ptrsize:8;"

    def jGetLoadedDynamicLibrariesInfos(self, packet):
        if "fetch_all_solibs" not in packet:
            return "OK"
        image = {
            "load_address": 0x180000000,
            "mod_date": 0,
            "pathname": DYLIB,
            # No UUID, so lldb looks the image up in the shared cache by name.
            "uuid": "",
            "min_version_os_name": "driverkit",
            "mach_header": {
                "magic": MH_MAGIC_64,
                "cputype": 16777228,
                "cpusubtype": 2,
                "filetype": 6,
                "flags": 0,
            },
            "segments": [
                {
                    "name": "__TEXT",
                    "vmaddr": 0x180000000,
                    "vmsize": 0x4000,
                    "fileoff": 0,
                    "filesize": 0x4000,
                    "maxprot": 5,
                }
            ],
        }
        return escape_binary(json.dumps({"images": [image]}))

    def vCont(self):
        return "vCont;"

    def setBreakpoint(self, packet):
        return ""


@skipUnlessDarwin
@skipIfRemote
@skipIf(archs=no_match(["arm64", "arm64e"]))
class TestSharedCacheRemount(GDBRemoteTestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        GDBRemoteTestBase.setUp(self)
        libdyld = ctypes.CDLL(None)
        if not hasattr(libdyld, "dyld_image_segment_data_4HWTrace"):
            self.skipTest("libdyld can't map inferior shared caches")
        caches = [
            os.path.join(d, CACHE_NAME)
            for d in CACHE_DIRS
            if os.path.exists(os.path.join(d, CACHE_NAME))
        ]
        if not caches:
            self.skipTest("no DriverKit shared cache on this host")
        self.system_cache = caches[0]

    def read_magic(self, target):
        """Returns the image's Mach-O magic, or None if it wasn't loaded."""
        module = target.FindModule(lldb.SBFileSpec(DYLIB))
        if not module.IsValid():
            return None
        data = module.FindSection("__TEXT").GetSectionData(0, 4)
        error = lldb.SBError()
        magic = data.GetUnsignedInt32(error, 0)
        self.assertSuccess(error)
        return magic

    def debug_session(self, cache_path, cache_uuid):
        # Each mock server takes a single connection. The first session uses
        # the one GDBRemoteTestBase.setUp started; later ones need their own.
        if getattr(self, "used_server", False):
            self.server.stop()
            self.server = MockGDBServer(self.server_socket_class())
            self.server.start()
        self.used_server = True
        self.server.responder = Responder(cache_path, cache_uuid)
        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        magic = self.read_magic(target)
        process.Kill()
        self.dbg.DeleteTarget(target)
        return magic

    def test_cache_on_unmounted_volume(self):
        """Debug with the cache on a disk image, force-detach the image, then
        debug again with the same cache at another path. lldb must neither
        crash nor use the module it made from the first copy. It doesn't open a
        cache with that UUID again, so with no memory to read from in this
        test, the image isn't loaded at all."""
        with open(self.system_cache, "rb") as f:
            cache_uuid = str(uuid.UUID(bytes=f.read(0x68)[0x58:0x68])).upper()

        # An earlier run that failed may have left its disk image attached.
        for device in attached_images_under(self.getBuildDir()):
            self.assertTrue(
                force_detach(device),
                "couldn't detach %s, left attached by an earlier run" % device,
            )

        # Names unique to this run, so it never reuses an earlier run's image.
        tag = uuid.uuid4().hex[:8]
        image = self.getBuildArtifact("cache-%s.dmg" % tag)
        mnt = self.getBuildArtifact("mnt-%s" % tag)
        os.makedirs(mnt)
        subprocess.check_call(
            ["hdiutil", "create", "-quiet", "-size", "64m", "-fs", "APFS"]
            + ["-volname", "LLDBSharedCache", image]
        )
        device = attach(image, mnt)
        self.assertTrue(device, "couldn't find the device of %s" % image)

        def detach_if_attached():
            if device in attached_images_under(self.getBuildDir()):
                self.assertTrue(force_detach(device), "couldn't detach %s" % device)

        self.addTearDownHook(detach_if_attached)
        mounted_cache = os.path.join(mnt, CACHE_NAME)
        shutil.copyfile(self.system_cache, mounted_cache)
        atlas = self.system_cache + ".atlas"
        if os.path.exists(atlas):
            shutil.copyfile(atlas, mounted_cache + ".atlas")

        self.assertEqual(self.debug_session(mounted_cache, cache_uuid), MH_MAGIC_64)

        # Pull the volume out from under the mapping lldb made of the cache,
        # then present the same cache again at a different path.
        self.assertTrue(force_detach(device), "couldn't detach %s" % device)
        remounted = self.getBuildArtifact("remounted")
        os.makedirs(remounted, exist_ok=True)
        remounted_cache = os.path.join(remounted, CACHE_NAME)
        shutil.copyfile(self.system_cache, remounted_cache)

        self.assertIn(
            self.debug_session(remounted_cache, cache_uuid), (None, MH_MAGIC_64)
        )
