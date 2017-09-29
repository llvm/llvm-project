; RUN: llc -mtriple=mipsel-linux-gnu -o /dev/null                 \
; RUN:     -verify-machineinstrs -stop-before=expand-isel-pseudos \
; RUN:     -debug-only=isel %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=mips64el-linux-gnu -o /dev/null               \
; RUN:     -verify-machineinstrs -stop-before=expand-isel-pseudos \
; RUN:     -debug-only=isel %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=mips-linux-gnu -o /dev/null                   \
; RUN:     -verify-machineinstrs -stop-before=expand-isel-pseudos \
; RUN:     -debug-only=isel %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=mips64-linux-gnu -o /dev/null                 \
; RUN:     -verify-machineinstrs -stop-before=expand-isel-pseudos \
; RUN:     -debug-only=isel %s 2>&1 | FileCheck %s


%struct.Str1 = type { [64 x i32] }

@s1 = common global %struct.Str1 zeroinitializer, align 4

define void @foo1() {
entry:
  call void @bar1(%struct.Str1* byval align 4 @s1)
  ret void
  ; CHECK-LABEL: *** MachineFunction at end of ISel ***
  ; CHECK-LABEL: # Machine code for function foo1: IsSSA, TracksLiveness
  ; CHECK: ADJCALLSTACKDOWN
  ; CHECK: JAL <es:memcpy>
  ; CHECK: ADJCALLSTACKUP
  ; CHECK: ADJCALLSTACKDOWN
  ; CHECK: JAL <ga:@bar1>
  ; CHECK: ADJCALLSTACKUP
  ; CHECK-LABEL: # End machine code for function foo1.
}

declare void @bar1(%struct.Str1* byval align 4)

define void @foo2() {
entry:
  call void @bar2(%struct.Str1* byval align 4 @s1, %struct.Str1* byval align 4 @s1)
  ret void
  ; CHECK-LABEL: *** MachineFunction at end of ISel ***
  ; CHECK-LABEL: # Machine code for function foo2: IsSSA, TracksLiveness
  ; CHECK: ADJCALLSTACKDOWN
  ; CHECK: JAL <es:memcpy>
  ; CHECK: ADJCALLSTACKUP
  ; CHECK: ADJCALLSTACKDOWN
  ; CHECK: JAL <es:memcpy>
  ; CHECK: ADJCALLSTACKUP
  ; CHECK: ADJCALLSTACKDOWN
  ; CHECK: JAL <ga:@bar2>
  ; CHECK: ADJCALLSTACKUP
  ; CHECK-LABEL: # End machine code for function foo2.
}

declare void @bar2(%struct.Str1* byval align 4, %struct.Str1* byval align 4)

