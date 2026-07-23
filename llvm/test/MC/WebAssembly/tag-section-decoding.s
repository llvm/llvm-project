# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling -filetype=obj %s | obj2yaml | FileCheck %s

# This is a regression test for a decoding bug that happens when a tag's
# sigindex is greater than 63, so we put 63 dummy functions with different
# signatures before the function that contains the 'throw' instruction to make
# the tag's sigindex 64.

.tagtype my_exception i32

.globl dummy0
dummy0:
  .functype dummy0 () -> (i32)
  i32.const 0
  end_function

.globl dummy1
dummy1:
  .functype dummy1 (i32) -> (i32)
  i32.const 0
  end_function

.globl dummy2
dummy2:
  .functype dummy2 (i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy3
dummy3:
  .functype dummy3 (i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy4
dummy4:
  .functype dummy4 (i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy5
dummy5:
  .functype dummy5 (i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy6
dummy6:
  .functype dummy6 (i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy7
dummy7:
  .functype dummy7 (i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy8
dummy8:
  .functype dummy8 (i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy9
dummy9:
  .functype dummy9 (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy10
dummy10:
  .functype dummy10 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy11
dummy11:
  .functype dummy11 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy12
dummy12:
  .functype dummy12 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy13
dummy13:
  .functype dummy13 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy14
dummy14:
  .functype dummy14 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy15
dummy15:
  .functype dummy15 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy16
dummy16:
  .functype dummy16 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy17
dummy17:
  .functype dummy17 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy18
dummy18:
  .functype dummy18 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy19
dummy19:
  .functype dummy19 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy20
dummy20:
  .functype dummy20 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy21
dummy21:
  .functype dummy21 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy22
dummy22:
  .functype dummy22 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy23
dummy23:
  .functype dummy23 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy24
dummy24:
  .functype dummy24 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy25
dummy25:
  .functype dummy25 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy26
dummy26:
  .functype dummy26 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy27
dummy27:
  .functype dummy27 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy28
dummy28:
  .functype dummy28 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy29
dummy29:
  .functype dummy29 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy30
dummy30:
  .functype dummy30 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy31
dummy31:
  .functype dummy31 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy32
dummy32:
  .functype dummy32 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy33
dummy33:
  .functype dummy33 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy34
dummy34:
  .functype dummy34 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy35
dummy35:
  .functype dummy35 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy36
dummy36:
  .functype dummy36 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy37
dummy37:
  .functype dummy37 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy38
dummy38:
  .functype dummy38 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy39
dummy39:
  .functype dummy39 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy40
dummy40:
  .functype dummy40 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy41
dummy41:
  .functype dummy41 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy42
dummy42:
  .functype dummy42 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy43
dummy43:
  .functype dummy43 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy44
dummy44:
  .functype dummy44 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy45
dummy45:
  .functype dummy45 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy46
dummy46:
  .functype dummy46 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy47
dummy47:
  .functype dummy47 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy48
dummy48:
  .functype dummy48 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy49
dummy49:
  .functype dummy49 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy50
dummy50:
  .functype dummy50 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy51
dummy51:
  .functype dummy51 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy52
dummy52:
  .functype dummy52 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy53
dummy53:
  .functype dummy53 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy54
dummy54:
  .functype dummy54 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy55
dummy55:
  .functype dummy55 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy56
dummy56:
  .functype dummy56 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy57
dummy57:
  .functype dummy57 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy58
dummy58:
  .functype dummy58 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy59
dummy59:
  .functype dummy59 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy60
dummy60:
  .functype dummy60 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy61
dummy61:
  .functype dummy61 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy62
dummy62:
  .functype dummy62 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl dummy63
dummy63:
  .functype dummy63 (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
  i32.const 0
  end_function

.globl test_throw

test_throw:
  .functype test_throw (i32) -> (i32)
  local.get 0
  throw my_exception
  end_function

my_exception:

# Checks to see if the tag index is correctly decoded in ULEB128. If it is
# decoded with LEB128, 64 will not be correctly decoded. 64 is the smallest
# number with which its LEB128 and ULEB128 encodings are different, because its
# 7th least significant bit is not 0.
# CHECK:      - Type:            TAG
# CHECK-NEXT:    TagTypes:        [ 64 ]
