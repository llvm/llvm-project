"""
Test things related to the fission debug information style where the main object
file contains a skeleton compile unit and the main debug info is in .dwo files.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


class ExecTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_zero_dwo_id(self):
        """
        Test that we can load a .o file that has a skeleton compile unit
        with a DWO ID of zero. We do this by hacking up the yaml to emit
        zero as a DWO ID is both the .o file and .dwo file. Then we make
        sure we can resolve something in the debug information to verify
        that we were able to load the .dwo file corrrectly since that is
        the only place that has this information.
        """
        src_dir = self.getSourceDir()
        dwo_yaml_path = os.path.join(src_dir, "main.dwo.yaml")
        obj_yaml_path = os.path.join(src_dir, "main.o.yaml")
        dwo_path = self.getBuildArtifact("main.dwo")
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(dwo_yaml_path, dwo_path)
        self.yaml2obj(obj_yaml_path, obj_path)

        # We need the current working directory to be set to the build directory
        os.chdir(self.getBuildDir())
        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        # Set a breakpoint by file and line, this doesn't require anything from
        # the .dwo file.
        bp = target.BreakpointCreateByLocation("main.cpp", 6)
        self.assertEqual(bp.GetNumLocations(), 1)
        bp_loc = bp.GetLocationAtIndex(0)
        self.assertTrue(bp_loc.IsValid())

        # We will use the address of the location to resolve the function "main"
        # to make sure we were able to open the .dwo file since this is the only
        # place that contains debug info for the function.
        self.assertTrue(bp_loc.GetAddress().GetFunction().IsValid())
