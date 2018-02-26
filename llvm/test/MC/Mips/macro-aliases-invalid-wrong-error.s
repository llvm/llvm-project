# RUN: not llvm-mc -arch=mips %s 2>%t1
# RUN: FileCheck --check-prefix=O32 %s < %t1

# RUN: not llvm-mc -arch=mips64 %s 2>%t1
# RUN: FileCheck --check-prefix=N64 %s < %t1

# Check that subu only rejects any non-constant values.

.globl end
  subu  $4, $4, %lo($start)   # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, $4, $start        # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, $a4, $a4          # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, $4, %hi(end)      # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, $4, end + 4       # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, $4, end           # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, $4, sp            # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list

  subu  $4, %lo($start)       # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, $start            # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, $a4               # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, %hi(end)          # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, end + 4           # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, end               # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
  subu  $4, sp                # O32: [[@LINE]]:{{[0-9]+}}: error: unexpected token in argument list
                              # N64: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list

$start:
