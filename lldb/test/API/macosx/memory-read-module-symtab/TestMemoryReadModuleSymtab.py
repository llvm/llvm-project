import time
import lldb
import glob
import re
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MemoryReadModuleSymtabTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def packet_log(self, name):
        """Enable packet logging to a scratch file, removed on teardown."""
        logfile = os.path.join(
            self.getBuildDir(), "%s-%s.txt" % (name, self.getArchitecture())
        )
        self.runCmd("log enable -f %s gdb-remote packets" % logfile)

        def cleanup():
            self.runCmd("log disable gdb-remote packets")
            if os.path.exists(logfile):
                os.unlink(logfile)

        self.addTearDownHook(cleanup)
        return logfile

    def memory_read_requests(self, logfile):
        """Scan a gdb-remote packet log for memory read requests,
        return an array of [start-address, end-address] entries."""
        self.assertTrue(os.path.exists(logfile), "packet log was created")
        log_text = [s.rstrip() for s in open(logfile).readlines()]
        read_lines = [line for line in log_text if "send packet: $x" in line]

        results = list()
        for l in read_lines:
            pattern = r"send packet: [$]x([0-9a-f]+),([0-9a-f]+)#"
            match = re.search(pattern, l)
            if match:
                start = int(match.group(1), 16)
                end = start + int(match.group(2), 16)
                entry = [start, end]
                results.append(entry)

        print(results)
        return results

    def confirm_datafilecache_dir_contents(self, cache_dir):
        # Check that we have a symtab created for a.out.
        module_file_glob = os.path.join(cache_dir, "llvmcache-*a.out*-symtab-*")
        self.assertEqual(len(glob.glob(module_file_glob)), 1)

        # Check that we don't have a symtab for a system library.
        module_file_glob = os.path.join(
            cache_dir, "llvmcache-*libsystem_c.dylib*-symtab-*"
        )
        self.assertEqual(len(glob.glob(module_file_glob)), 0)

    def attach_to_process(self, pid):
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid(), "Got a vaid empty target.")
        error = lldb.SBError()
        attach_info = lldb.SBAttachInfo()
        attach_info.SetProcessID(pid)
        attach_info.SetIgnoreExisting(False)
        process = target.Attach(attach_info, error)
        self.assertSuccess(error, "Didn't attach successfully to %d" % (pid))
        return target, process

    @requireDarwin
    def test_memory_read_module_symtab(self):
        """Attach to a process where no on-disk copy of the binary exists.
        On first attach, we create a DataFileCache SymbolTable for the
        binary.
        Detach, delete the target, flush the module cache.
        On second attach, we turn on packet communication logging,
        attach to the process again, and should get the symbol table
        from the DataFileCache.
        Check the packet log to ensure that no reads were made of the
        binary's symbol table."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Use a file as a synchronization point between test and inferior: the
        # inferior writes its pid only after it has called PT_DENY_ATTACH.
        pid_file_path = lldbutil.append_to_process_working_directory(
            self, "pid_file_%d" % (int(time.time()))
        )
        self.addTearDownHook(
            lambda: self.run_platform_command("rm %s" % (pid_file_path))
        )

        cache_dir = os.path.join(self.getBuildDir(), "lldb-module-cache")
        self.runCmd('settings set -g symbols.lldb-index-cache-path "%s"' % (cache_dir))
        self.runCmd("settings set -g symbols.enable-lldb-index-cache false")
        self.runCmd(
            "settings set -g symbols.enable-lldb-index-cache-memory-modules true"
        )

        # Launch the process, so it can remove the binary before
        # lldb attaches.
        popen = self.spawnSubprocess(exe, [pid_file_path])
        pid = int(lldbutil.wait_for_file_on_target(self, pid_file_path))

        # First attach:  Create the DataFileCache for the MemoryModule.
        target, process = self.attach_to_process(pid)
        self.confirm_datafilecache_dir_contents(cache_dir)
        process.Detach()

        # Delete the target, flush the Module we've created, so
        # we'll need to re-create it.
        self.dbg.DeleteTarget(target)
        self.assertEqual(self.dbg.GetNumTargets(), 0)
        self.dbg.MemoryPressureDetected()

        # Enable logging of gdb-remote packets, to look for any
        # reads of a.out's __LINKEDIT in this second attach.
        logfile = self.packet_log("packets")

        # Second attach:  Read the DataFileCache for the MemoryModule.
        target, process = self.attach_to_process(pid)
        self.confirm_datafilecache_dir_contents(cache_dir)

        # Ensure we've parsed the a.out Module.
        self.runCmd("bt")

        exe_filespec = lldb.SBFileSpec("a.out")
        exe_module = target.FindModule(exe_filespec)
        print(exe_module)
        exe_linkedit = exe_module.FindSection("__LINKEDIT")
        print(exe_linkedit)
        self.assertTrue(exe_linkedit.IsValid())
        linkedit_startaddr = exe_linkedit.GetLoadAddress(target)
        linkedit_endaddr = linkedit_startaddr + exe_linkedit.GetByteSize()

        memory_read_addrs = self.memory_read_requests(logfile)
        self.assertGreater(len(memory_read_addrs), 0)

        # Iterate through the start & end address of every memory
        # read packet, and confirm that none of them are within
        # the VM range of a.out's __LINKEDIT.  The SymbolTable for
        # a.out should be gotten from the DataFileCache, not memory.
        for addrs in memory_read_addrs:
            for addr in addrs:
                if addr >= linkedit_startaddr and addr < linkedit_endaddr:
                    print(
                        "Address 0x%x was read from a.out's __LINKEDIT; all symbol table information should have been retrieved from DataFileCache of it."
                        % addr
                    )
                    if self.TraceOn():
                        self.runCmd("target modules dump sections a.out")
                        print("Read of address 0x%x is within LINKEDIT" % addr)
                self.assertFalse(addr >= linkedit_startaddr and addr < linkedit_endaddr)
