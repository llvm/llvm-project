"""
Test that SBValue doesn't incorrectly sign-extend
the Scalar value of a bitfield that has an unsigned
enum type.

We test this by assigning to a bit-field a value
that is out-of-range of it's signed counterpart.
I.e., with a bit-field of width 4, assigning
8 to it would be out-of-range if we treated it
as a signed. If LLDB were to sign-extend the Scalar
(which shouldn't happen for unsigned bit-fields)
it would left-fill the result with 1s; we test
for this not to happen.
"""

import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(__file__, globals(), [skipIf(dwarf_version=["<", "3"])])
