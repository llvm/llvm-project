// RUN: llvm-mc < %s -triple armv8a -filetype=obj | \
// RUN:   llvm-objdump --triple armv8a -d --visualize-jumps=unicode - | \
// RUN:   diff - %p/Inputs/visualize-jumps-arm-unicode.txt

// RUN: llvm-mc < %s -triple armv8a -filetype=obj | \
// RUN:   llvm-objdump --triple armv8a -d --visualize-jumps=ascii - | \
// RUN:   diff - %p/Inputs/visualize-jumps-arm-ascii.txt

test_func:
  // Relocated instructions don't get control-flow edges.
  bl extern_func
  b extern_func
  
  // Two branches to the same label, one forward and one backward.
  b .Llabel1
.Llabel1:
  nop
  b .Llabel1

  // Branch to self, no CFG edge shown
  b .

  // Conditional branches
  beq .Llabel2
.Llabel2:
  nop

  // Branches are sorted with shorter ones to the right, to reduce number of
  // crossings, and keep the lines for short branches short themselves.
  b .Llabel5
  b .Llabel4
  b .Llabel3
.Llabel3:
  nop
.Llabel4:
  nop
.Llabel5:
  nop

  // Sometimes crossings can't be avoided.
  b .Llabel6
  b .Llabel7
.Llabel6:
  nop
.Llabel7:
  nop

  // TODO If a branch goes to another branch instruction, we don't have a way
  // to represent that. Can we improve on this?
  b .Llabel8
.Llabel8:
  b .Llabel9
.Llabel9:
  nop
