

import gdbremote_testcase
import textwrap
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import re
import xml.etree.ElementTree as ET

class TestGdbRemoteTargetXmlPacket(gdbremote_testcase.GdbRemoteTestCaseBase):

    @llgs_test
    def test_g_target_xml_returns_correct_data(self):
        self.build()
        self.set_inferior_startup_launch()

        procs = self.prep_debug_monitor_and_inferior()

        OFFSET = 0
        LENGTH = 0x1ffff0
        self.test_sequence.add_log_lines([
            "read packet: $qXfer:features:read:target.xml:{:x},{:x}#00".format(
                    OFFSET,
                    LENGTH),
            {   
                "direction": "send", 
                "regex": re.compile("^\$l(.+)#[0-9a-fA-F]{2}$"), 
                "capture": {1: "target_xml"}
            }],
            True)
        context = self.expect_gdbremote_sequence()

        target_xml = context.get("target_xml")
        
        root = ET.fromstring(target_xml)
        self.assertIsNotNone(root)
        self.assertEqual(root.tag, "target")

        architecture = root.find("architecture")
        self.assertIsNotNone(architecture)
        self.assertIn(self.getArchitecture(), architecture.text)

        feature = root.find("feature")
        self.assertIsNotNone(feature)

        target_xml_registers = feature.findall("reg")
        self.assertTrue(len(target_xml_registers) > 0)

        # registers info collected by qRegisterInfo
        self.add_register_info_collection_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        q_info_registers = self.parse_register_info_packets(context)

        self.assertEqual(len(target_xml_registers), len(q_info_registers))
        for register in zip(target_xml_registers, q_info_registers):
            xml_info_reg = register[0]
            q_info_reg = register[1]
            self.assertEqual(q_info_reg["name"], xml_info_reg.get("name"))
            self.assertEqual(q_info_reg["set"], xml_info_reg.get("group"))
            self.assertEqual(q_info_reg["format"], xml_info_reg.get("format"))
            self.assertEqual(q_info_reg["bitsize"], xml_info_reg.get("bitsize"))

            if not self.isAArch64():
                self.assertEqual(q_info_reg["offset"], xml_info_reg.get("offset"))

            self.assertEqual(q_info_reg["encoding"], xml_info_reg.get("encoding"))
