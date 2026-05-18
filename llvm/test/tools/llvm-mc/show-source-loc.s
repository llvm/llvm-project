# RUN: llvm-mc -triple=x86_64 -show-source-loc %s -I %S/Inputs | FileCheck %s

## Check that -show-source-loc emits <SourceLoc: ...> comments after instructions.

.macro inner_macro reg1, reg2
  addl \reg1, \reg2
  subl \reg2, \reg1
.endm

.macro outer_macro r1, r2
  inner_macro \r1, \r2
.endm

## Standard instructions report their exact line number.
nop
# CHECK: nop
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-2]]:1>

## Nested macro expansion reports the full expansion stack.
outer_macro %eax, %ebx
# CHECK: addl %eax, %ebx
# CHECK-NEXT: # <ExpansionLoc: <instantiation>:1:1>
# CHECK-NEXT: # <ExpansionLoc: <instantiation>:1:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-4]]:1>
# CHECK: subl %ebx, %eax
# CHECK-NEXT: # <ExpansionLoc: <instantiation>:2:3>
# CHECK-NEXT: # <ExpansionLoc: <instantiation>:1:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-8]]:1>

## .include reports the include file location and the .include directive location.
.include "show-source-loc.inc"
# CHECK: xorl %ecx, %ecx
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.inc:1:1>
# CHECK-NEXT: # <IncludeLoc: {{.*}}show-source-loc.s:[[#@LINE-3]]:31>

## Macro defined in an include file.
inc_macro %edx
# CHECK: incl %edx
# CHECK-NEXT: # <ExpansionLoc: <instantiation>:1:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-3]]:1>

## Indefinite repeat block (.irp).
.irp reg, %esi, %edi
  incl \reg
.endr
# CHECK: incl %esi
# CHECK-NEXT: # <ExpansionLoc: <instantiation>:1:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-5]]:1>
# CHECK: incl %edi
# CHECK-NEXT: # <ExpansionLoc: <instantiation>:2:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-8]]:1>
