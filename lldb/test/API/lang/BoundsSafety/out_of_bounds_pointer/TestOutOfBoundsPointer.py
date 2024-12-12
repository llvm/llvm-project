import lldb
import re
from lldbsuite.test.decorators import *
from lldbsuite.test.decorators import _get_bool_config
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from enum import Enum


class OOBKind(Enum):
    Full = (0,)
    Partial = (1,)
    Overflow = (2,)
    NotOOB = 3


def OOBKindToStr(kind: OOBKind) -> str:
    if kind is OOBKind.Full:
        return "out-of-bounds"
    if kind is OOBKind.Partial:
        return "partially out-of-bounds"
    if kind is OOBKind.Overflow:
        return "overflown bounds"
    if kind is OOBKind.NotOOB:
        return ""


class TestOutOfBoundsPointer(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @staticmethod
    def get_bidi_var_regex(oob_kind, type_name):
        if oob_kind == OOBKind.NotOOB:
            return (
                r"\({type_name} *__bidi_indexable\) [a-zA-Z_0-9]+ = "
                "\(ptr: 0x[a-f0-9]+, bounds: 0x[a-f0-9]+..0x[a-f0-9]+\)"
            ).format(type_name=re.escape(type_name))
        else:
            oob = OOBKindToStr(oob_kind)
            return (
                r"\({type_name} *__bidi_indexable\) [a-zA-Z_0-9]+ = "
                "\({oob} ptr: 0x[a-f0-9]+..0x[a-f0-9]+, bounds: 0x[a-f0-9]+..0x[a-f0-9]+\)"
            ).format(type_name=re.escape(type_name), oob=re.escape(oob))

    @staticmethod
    def get_idx_var_regex(oob_kind, type_name):
        if oob_kind == OOBKind.NotOOB:
            return (
                r"\({type_name} *__indexable\) [a-zA-Z_0-9]+ = "
                "\(ptr: 0x[a-f0-9]+, upper bound: 0x[a-f0-9]+\)"
            ).format(type_name=re.escape(type_name))
        else:
            oob = OOBKindToStr(oob_kind)
            return (
                r"\({type_name} *__indexable\) [a-zA-Z_0-9]+ = "
                "\({oob} ptr: 0x[a-f0-9]+..0x[a-f0-9]+, upper bound: 0x[a-f0-9]+\)"
            ).format(type_name=re.escape(type_name), oob=re.escape(oob))

    def bidi_in_bounds(self, type_name):
        return self.get_bidi_var_regex(oob_kind=OOBKind.NotOOB, type_name=type_name)

    def bidi_full_oob(self, type_name):
        return self.get_bidi_var_regex(oob_kind=OOBKind.Full, type_name=type_name)

    def bidi_partial_oob(self, type_name):
        return self.get_bidi_var_regex(oob_kind=OOBKind.Partial, type_name=type_name)

    def bidi_overflow_oob(self, type_name):
        return self.get_bidi_var_regex(oob_kind=OOBKind.Overflow, type_name=type_name)

    def in_bounds(self, type_name):
        return self.get_idx_var_regex(oob_kind=OOBKind.NotOOB, type_name=type_name)

    def full_oob(self, type_name):
        return self.get_idx_var_regex(oob_kind=OOBKind.Full, type_name=type_name)

    def partial_oob(self, type_name):
        return self.get_idx_var_regex(oob_kind=OOBKind.Partial, type_name=type_name)

    def overflow_oob(self, type_name):
        return self.get_idx_var_regex(oob_kind=OOBKind.Overflow, type_name=type_name)

    def test_bidi_known_type_size(self):
        self.build()

        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, r"// break here:.+", lldb.SBFileSpec("bidi_check_known_type_size.c")
        )

        # -----------------------------------------------------------------------
        # ptr < lower bound
        # -----------------------------------------------------------------------
        self.expect(
            "frame variable oob_ptr_lower", patterns=[self.bidi_in_bounds("char *")]
        )
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_ptr_lower", patterns=[self.bidi_full_oob("char *")]
        )

        # -----------------------------------------------------------------------
        # ptr + type size overflows
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_ptr_plus_size_overflows",
            patterns=[self.bidi_overflow_oob("int *")],
        )

        # -----------------------------------------------------------------------
        # ptr >= upper bound
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_upper", patterns=[self.bidi_in_bounds("char *")]
        )
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_upper", patterns=[self.bidi_in_bounds("char *")]
        )
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_upper", patterns=[self.bidi_in_bounds("char *")]
        )
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.bidi_full_oob("char *")])

        # -----------------------------------------------------------------------
        # Upper bound not aligned with type
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_upper_type_pun",
            patterns=[self.bidi_partial_oob("int *")],
        )

        # -----------------------------------------------------------------------
        # nullptr
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_null", patterns=[self.bidi_full_oob("int *")])

        # -----------------------------------------------------------------------
        # Flexible array member
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable fams0", patterns=[self.bidi_in_bounds("FAMS_t *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable fams1", patterns=[self.bidi_in_bounds("FAMS_t *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable fams2", patterns=[self.bidi_full_oob("FAMS_t *")])

    def test_bidi_unknown_type_size(self):
        self.build()

        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, r"// break here:.+", lldb.SBFileSpec("bidi_check_unknown_type_size.c")
        )

        # -----------------------------------------------------------------------
        # ptr < lower bound
        # -----------------------------------------------------------------------
        self.expect(
            "frame variable oob_ptr_lower", patterns=[self.bidi_in_bounds("void *")]
        )
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_ptr_lower", patterns=[self.bidi_full_oob("void *")]
        )

        # -----------------------------------------------------------------------
        # ptr + type size overflows
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_ptr_plus_size_overflows",
            patterns=[self.bidi_overflow_oob("void *")],
        )

        # -----------------------------------------------------------------------
        # ptr >= upper bound
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_upper", patterns=[self.bidi_in_bounds("void *")]
        )
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_upper", patterns=[self.bidi_in_bounds("void *")]
        )
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_upper", patterns=[self.bidi_in_bounds("void *")]
        )
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.bidi_full_oob("void *")])

        # -----------------------------------------------------------------------
        # nullptr
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_null", patterns=[self.bidi_full_oob("void *")])

    def test_idx_known_type_size(self):
        self.build()

        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, r"// break here:.+", lldb.SBFileSpec("idx_check_known_type_size.c")
        )

        # -----------------------------------------------------------------------
        # ptr + type size overflows
        # -----------------------------------------------------------------------
        self.expect(
            "frame variable oob_ptr_plus_size_overflows",
            patterns=[self.overflow_oob("int *")],
        )

        # -----------------------------------------------------------------------
        # ptr > upper bound
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.in_bounds("char *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.in_bounds("char *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.in_bounds("char *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.full_oob("char *")])

        # -----------------------------------------------------------------------
        # Upper bound not aligned with type
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect(
            "frame variable oob_upper_type_pun", patterns=[self.partial_oob("int *")]
        )

        # -----------------------------------------------------------------------
        # nullptr
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_null", patterns=[self.full_oob("int *")])

        # -----------------------------------------------------------------------
        # Flexible array member
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable fams0", patterns=[self.in_bounds("FAMS_t *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable fams1", patterns=[self.in_bounds("FAMS_t *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable fams2", patterns=[self.full_oob("FAMS_t *")])

    def test_idx_unknown_type_size(self):
        self.build()

        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, r"// break here:.+", lldb.SBFileSpec("idx_check_unknown_type_size.c")
        )

        # -----------------------------------------------------------------------
        # ptr + type size overflows
        # -----------------------------------------------------------------------
        self.expect(
            "frame variable oob_ptr_plus_size_overflows",
            patterns=[self.overflow_oob("void *")],
        )

        # -----------------------------------------------------------------------
        # ptr >= upper bound
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.in_bounds("void *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.in_bounds("void *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.in_bounds("void *")])
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_upper", patterns=[self.full_oob("void *")])

        # -----------------------------------------------------------------------
        # nullptr
        # -----------------------------------------------------------------------
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect("frame variable oob_null", patterns=[self.full_oob("void *")])
