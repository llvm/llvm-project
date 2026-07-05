from dap_server import Source
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os
import lldb


class TestDAP_InstructionBreakpointTestCase(lldbdap_testcase.DAPTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        lldbdap_testcase.DAPTestCaseBase.setUp(self)

        self.main_basename = "main-copy.cpp"
        self.main_path = os.path.realpath(self.getBuildArtifact(self.main_basename))

    @skipIfWindows
    def test_instruction_breakpoint(self):
        self.build()
        self.instruction_breakpoint_test()

    def instruction_breakpoint_test(self):
        """Sample test to ensure SBFrame::Disassemble produces SOME output"""
        # Create a target by the debugger.
        target = self.createTestTarget()

        main_line = line_number("main.cpp", "breakpoint 1")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set source breakpoint 1
        response = self.dap_server.request_setBreakpoints(
            Source.build(path=self.main_path), [main_line]
        )
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 1)
        breakpoint = breakpoints[0]
        self.assertEqual(
            breakpoint["line"], main_line, "incorrect breakpoint source line"
        )
        self.assertTrue(breakpoint["verified"], "breakpoint is not verified")
        self.assertEqual(
            self.main_basename, breakpoint["source"]["name"], "incorrect source name"
        )
        self.assertEqual(
            self.main_path, breakpoint["source"]["path"], "incorrect source file path"
        )
        other_breakpoint_id = breakpoint["id"]

        # Continue and then verifiy the breakpoint
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([other_breakpoint_id])

        # now we check the stack trace making sure that we got mapped source paths
        frames = self.dap_server.request_stackTrace()["body"]["stackFrames"]
        intstructionPointerReference = []
        setIntstructionBreakpoints = []
        intstructionPointerReference.append(frames[0]["instructionPointerReference"])
        self.assertEqual(
            frames[0]["source"]["name"], self.main_basename, "incorrect source name"
        )
        self.assertEqual(
            frames[0]["source"]["path"], self.main_path, "incorrect source file path"
        )

        # Check disassembly view
        disassembled_instructions, instruction = self.disassemble(frameIndex=0)
        self.assertEqual(
            instruction["address"],
            intstructionPointerReference[0],
            "current breakpoint reference is not in the disaasembly view",
        )

        # Get next instruction address to set instruction breakpoint
        instruction_addr_list = list(disassembled_instructions.keys())
        index = instruction_addr_list.index(intstructionPointerReference[0])
        if len(instruction_addr_list) >= (index + 1):
            next_inst_addr = instruction_addr_list[index + 1]
            if len(next_inst_addr) > 2:
                setIntstructionBreakpoints.append(next_inst_addr)
                instruction_breakpoint_response = (
                    self.dap_server.request_setInstructionBreakpoints(
                        setIntstructionBreakpoints
                    )
                )
                inst_breakpoints = instruction_breakpoint_response["body"][
                    "breakpoints"
                ]
                self.assertEqual(
                    inst_breakpoints[0]["instructionReference"],
                    next_inst_addr,
                    "Instruction breakpoint has not been resolved or failed to relocate the instruction breakpoint",
                )
                self.dap_server.request_continue()
                self.verify_breakpoint_hit([inst_breakpoints[0]["id"]])
