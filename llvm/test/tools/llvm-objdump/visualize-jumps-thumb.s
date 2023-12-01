// RUN: llvm-mc < %s -triple thumbv8a -filetype=obj | \
// RUN:   llvm-objdump --triple thumbv8a -d --visualize-jumps=unicode - | \
// RUN:   diff - %p/Inputs/visualize-jumps-thumb-unicode.txt

// RUN: llvm-mc < %s -triple thumbv8a -filetype=obj | \
// RUN:   llvm-objdump --triple thumbv8a -d --visualize-jumps=ascii - | \
// RUN:   diff - %p/Inputs/visualize-jumps-thumb-ascii.txt

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

  // Different branch instructions
  b.w .Llabel2
  beq .Llabel2
  bne.w .Llabel2
  cbz r0, .Llabel2
  it le
  ble .Llabel2
  nop
.Llabel2:
  nop
.Llabel2.1:
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

