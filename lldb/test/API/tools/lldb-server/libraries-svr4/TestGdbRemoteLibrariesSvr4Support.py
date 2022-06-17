import xml.etree.ElementTree as ET
import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class TestGdbRemoteLibrariesSvr4Support(gdbremote_testcase.GdbRemoteTestCaseBase):

    FEATURE_NAME = "qXfer:libraries-svr4:read"

    def setup_test(self):
        self.build()
        self.set_inferior_startup_launch()
        env = {}
        env[self.dylibPath] = self.getBuildDir()
        self.prep_debug_monitor_and_inferior(inferior_env=env)
        self.continue_process_and_wait_for_stop()

    def get_expected_libs(self):
        return ["libsvr4lib_a.so", 'libsvr4lib_b".so']

    def has_libraries_svr4_support(self):
        self.add_qSupported_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        features = self.parse_qSupported_response(context)
        return self.FEATURE_NAME in features and features[self.FEATURE_NAME] == "+"

    def get_libraries_svr4_data(self):
        # Start up llgs and inferior, and check for libraries-svr4 support.
        if not self.has_libraries_svr4_support():
            self.skipTest("libraries-svr4 not supported")

        # Grab the libraries-svr4 data.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $qXfer:libraries-svr4:read::0,ffff:#00",
                {
                    "direction": "send",
                    "regex": re.compile(
                        r"^\$([^E])(.*)#[0-9a-fA-F]{2}$", re.MULTILINE | re.DOTALL
                    ),
                    "capture": {1: "response_type", 2: "content_raw"},
                },
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure we end up with all libraries-svr4 data in one packet.
        self.assertEqual(context.get("response_type"), "l")

        # Decode binary data.
        content_raw = context.get("content_raw")
        self.assertIsNotNone(content_raw)
        return content_raw

    def get_libraries_svr4_xml(self):
        libraries_svr4 = self.get_libraries_svr4_data()
        xml_root = None
        try:
            xml_root = ET.fromstring(libraries_svr4)
        except xml.etree.ElementTree.ParseError:
            pass
        self.assertIsNotNone(xml_root, "Malformed libraries-svr4 XML")
        return xml_root

    def libraries_svr4_well_formed(self):
        xml_root = self.get_libraries_svr4_xml()
        self.assertEqual(xml_root.tag, "library-list-svr4")
        for child in xml_root:
            self.assertEqual(child.tag, "library")
            self.assertItemsEqual(child.attrib.keys(), ["name", "lm", "l_addr", "l_ld"])

    def libraries_svr4_has_correct_load_addr(self):
        xml_root = self.get_libraries_svr4_xml()
        for child in xml_root:
            name = child.attrib.get("name")
            base_name = os.path.basename(name)
            if os.path.basename(name) not in self.get_expected_libs():
                continue
            load_addr = int(child.attrib.get("l_addr"), 16)
            self.reset_test_sequence()
            self.add_query_memory_region_packets(load_addr)
            context = self.expect_gdbremote_sequence()
            mem_region = self.parse_memory_region_packet(context)
            self.assertEqual(load_addr, int(mem_region.get("start", 0), 16))
            self.assertEqual(
                os.path.realpath(name), os.path.realpath(mem_region.get("name", ""))
            )

    def libraries_svr4_libs_present(self):
        xml_root = self.get_libraries_svr4_xml()
        libraries_svr4_names = []
        for child in xml_root:
            name = child.attrib.get("name")
            libraries_svr4_names.append(os.path.realpath(name))
        for lib in self.get_expected_libs():
            self.assertIn(os.path.realpath(self.getBuildDir() + "/" + lib), libraries_svr4_names)

    @skipUnlessPlatform(["linux", "android", "freebsd", "netbsd"])
    def test_supports_libraries_svr4(self):
        self.setup_test()
        self.assertTrue(self.has_libraries_svr4_support())

    @skipUnlessPlatform(["linux", "android", "freebsd", "netbsd"])
    @expectedFailureNetBSD
    def test_libraries_svr4_well_formed(self):
        self.setup_test()
        self.libraries_svr4_well_formed()

    @skipUnlessPlatform(["linux", "android", "freebsd", "netbsd"])
    @expectedFailureNetBSD
    def test_libraries_svr4_load_addr(self):
        self.setup_test()
        self.libraries_svr4_has_correct_load_addr()

    @skipUnlessPlatform(["linux", "android", "freebsd", "netbsd"])
    @expectedFailureNetBSD
    def test_libraries_svr4_libs_present(self):
        self.setup_test()
        self.libraries_svr4_libs_present()
