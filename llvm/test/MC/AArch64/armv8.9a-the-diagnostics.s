// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+the -mattr=+d128 < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-ZXR %s

rcwswpp   xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwswppa  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwswppal xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwswppl  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwswpp   x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwswppa  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwswppal x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwswppl  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction

rcwclrp   xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwclrpa  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwclrpal xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwclrpl  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwclrp   x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwclrpa  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwclrpal x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwclrpl  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction

rcwsetp   xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsetpa  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsetpal xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsetpl  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsetp   x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsetpa  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsetpal x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsetpl  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction

rcwsswpp   xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsswppa  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsswppal xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsswppl  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsswpp   x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsswppa  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsswppal x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsswppl  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction

rcwsclrp   xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsclrpa  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsclrpal xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsclrpl  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsclrp   x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsclrpa  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsclrpal x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwsclrpl  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction

rcwssetp   xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwssetpa  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwssetpal xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwssetpl  xzr, x5, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwssetp   x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwssetpa  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwssetpal x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
rcwssetpl  x5, xzr, [x4]
// ERROR-NO-ZXR:   error: invalid operand for instruction
