# RUN: llvm-mc -triple=x86_64 -show-source-loc %s -I %S/Inputs
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
# CHECK-NEXT: # <MacroLoc: {{.*}}show-source-loc.s:6:1>
# CHECK-NEXT: # <MacroLoc: {{.*}}show-source-loc.s:11:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-4]]:1>
# CHECK: subl %ebx, %eax
# CHECK-NEXT: # <MacroLoc: {{.*}}show-source-loc.s:6:1>
# CHECK-NEXT: # <MacroLoc: {{.*}}show-source-loc.s:11:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-8]]:1>

## .include reports the include file location and the .include directive location.
.include "show-source-loc.inc"
# CHECK: xorl %ecx, %ecx
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.inc:1:1>
# CHECK-NEXT: # <IncludeLoc: {{.*}}show-source-loc.s:[[#@LINE-3]]:31>

## Macro defined in an include file.
inc_macro %edx
# CHECK: incl %edx
# CHECK-NEXT: # <MacroLoc: {{.*}}show-source-loc.inc:3:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-3]]:1>

## Indefinite repeat block (.irp).
.irp reg, %esi, %edi
  incl \reg
.endr
# CHECK: incl %esi
# CHECK-NEXT: # <MacroLoc: {{.*}}show-source-loc.s:44:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-5]]:1>
# CHECK: incl %edi
# CHECK-NEXT: # <MacroLoc: {{.*}}show-source-loc.s:44:1>
# CHECK-NEXT: # <SourceLoc: {{.*}}show-source-loc.s:[[#@LINE-8]]:1>
