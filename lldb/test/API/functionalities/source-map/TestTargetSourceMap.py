import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import os


class TestTargetSourceMap(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_source_map(self):
        """Test target.source-map' functionality."""

        def assertBreakpointWithSourceMap(src_path):
            # Set a breakpoint after we remap source and verify that it succeeds
            bp = target.BreakpointCreateByLocation(src_path, 2)
            self.assertEquals(bp.GetNumLocations(), 1,
                            "make sure breakpoint was resolved with map")

            # Now make sure that we can actually FIND the source file using this
            # remapping:
            retval = lldb.SBCommandReturnObject()
            self.dbg.GetCommandInterpreter().HandleCommand("source list -f main.c -l 2", retval)
            self.assertTrue(retval.Succeeded(), "source list didn't succeed.")
            self.assertNotEqual(retval.GetOutput(), None, "We got no ouput from source list")
            self.assertTrue("return" in retval.GetOutput(), "We didn't find the source file...")

        # Set the target soure map to map "./" to the current test directory
        src_dir = self.getSourceDir()
        src_path = os.path.join(src_dir, "main.c")
        yaml_path = os.path.join(src_dir, "a.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)

        # Set a breakpoint before we remap source and verify that it fails
        bp = target.BreakpointCreateByLocation(src_path, 2)
        self.assertEquals(bp.GetNumLocations(), 0,
                        "make sure no breakpoints were resolved without map")

        invalid_path = src_dir + "invalid_path"
        invalid_path2 = src_dir + "invalid_path2"

        # We make sure the error message contains all the invalid paths
        self.expect(
            'settings set target.source-map . "%s" . "%s" . "%s"' % (invalid_path, src_dir, invalid_path2),
            substrs=[
                'the replacement path doesn\'t exist: "%s"' % (invalid_path),
                'the replacement path doesn\'t exist: "%s"' % (invalid_path2),
            ],
            error=True,
        )
        self.expect(
            'settings show target.source-map',
            substrs=['[0] "." -> "%s"' % (src_dir)],
        )
        assertBreakpointWithSourceMap(src_path)

        # Index 0 is the valid mapping, and modifying it to an invalid one should have no effect
        self.expect(
            'settings replace target.source-map 0 . "%s"' % (invalid_path),
            substrs=['error: the replacement path doesn\'t exist: "%s"' % (invalid_path)],
            error=True,
        )
        self.expect(
            'settings show target.source-map',
            substrs=['[0] "." -> "%s"' % (src_dir)]
        )
        assertBreakpointWithSourceMap(src_path)

        # Let's clear and add the mapping in with insert-after
        self.runCmd('settings remove target.source-map 0')
        self.expect(
            'settings show target.source-map',
            endstr="target.source-map (path-map) =\n",
        )
        
        # We add a valid but useless mapping so that we can use insert-after
        another_valid_path = os.path.dirname(src_dir)
        self.runCmd('settings set target.source-map . "%s"' % (another_valid_path))

        self.expect(
            'settings replace target.source-map 0 . "%s"' % (invalid_path),
            substrs=['error: the replacement path doesn\'t exist: "%s"' % (invalid_path)],
            error=True,
        )
        self.expect(
            'settings show target.source-map',
            substrs=['[0] "." -> "%s"' % (another_valid_path)]
        )

        # Let's clear and add the mapping in with append
        self.expect('settings remove target.source-map 0')
        self.expect(
            'settings show target.source-map',
            endstr="target.source-map (path-map) =\n",
        )

        self.expect(
            'settings append target.source-map . "%s" . "%s"' % (invalid_path, src_dir),
            substrs=['error: the replacement path doesn\'t exist: "%s"' % (invalid_path)],
            error=True,
        )
        assertBreakpointWithSourceMap(src_path)
