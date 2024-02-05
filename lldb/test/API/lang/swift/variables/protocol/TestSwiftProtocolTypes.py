# TestSwiftProtocolTypes.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test support for protocol types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftProtocolTypes(TestBase):
    @swiftTest
    def test_swift_protocol_types(self):
        """Test support for protocol types"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'Set breakpoint here',
                                          lldb.SBFileSpec('main.swift'))

        self.expect("frame variable --dynamic-type no-dynamic-values"
                    " --raw-output --show-types loc2d",
                    substrs=['PointUtils) loc2d =',
                             '(Builtin.RawPointer) payload_data_0 = 0x',
                             '(Builtin.RawPointer) payload_data_1 = 0x',
                             '(Builtin.RawPointer) payload_data_2 = 0x',
                             '(Any.Type) metadata = 0x',
                             '(Builtin.RawPointer) wtable = 0x'])
 
        self.expect("frame variable loc2d",
                    substrs=['Point2D) loc2d =',
                             'x = 1.25', 'y = 2.5'])
 
        self.expect("frame variable --dynamic-type no-dynamic-values"
                    " --raw-output --show-types loc3d",
                    substrs=['PointUtils) loc3d =',
                             '(Builtin.RawPointer) payload_data_0 = 0x',
                             '(Builtin.RawPointer) payload_data_1 = 0x',
                             '(Builtin.RawPointer) payload_data_2 = 0x',
                             '(Any.Type) metadata = 0x',
                             '(Builtin.RawPointer) wtable = 0x'])
 
        self.expect(
            "frame variable loc3d",
            substrs=[
                'Point3D) loc3d = 0x',
                'x = 1.25',
                'y = 2.5',
                'z = 1.25'])
 
        self.expect("expression --dynamic-type no-dynamic-values"
                    " --raw-output --show-types -- loc2d",
                    substrs=['PointUtils) $R',
                             '(Builtin.RawPointer) payload_data_0 = 0x',
                             '(Builtin.RawPointer) payload_data_1 = 0x',
                             '(Builtin.RawPointer) payload_data_2 = 0x',
                             '(Any.Type) metadata = 0x',
                             '(Builtin.RawPointer) wtable = 0x'])
 
        self.expect("expression -- loc2d",
                    substrs=['Point2D) $R',
                             'x = 1.25', 'y = 2.5'])
 
        self.expect("expression --dynamic-type no-dynamic-values"
                    " --raw-output --show-types -- loc3dCB",
                    substrs=['PointUtils & Swift.AnyObject) $R',
                             '(Builtin.RawPointer) object = 0x',
                             '(Builtin.RawPointer) wtable = 0x'])
 
        self.expect("expression -- loc3dCB",
                    substrs=['Point3D) $R', 'x = 1.25', 'y = 2.5', 'z = 1.25'])

        self.expect("expression --dynamic-type no-dynamic-values"
                    " --raw-output --show-types -- loc3dSuper",
                    substrs=['(a.PointSuperclass & a.PointUtils) $R',
                             '(a.PointSuperclass) object = 0x',
                             '(Swift.Int) superData = ',
                             '(Builtin.RawPointer) wtable = 0x'])

        self.expect("expression -- loc3dSuper",
                    substrs=['Point3D) $R', 'x = 1.25', 'y = 2.5', 'z = 1.25'])

