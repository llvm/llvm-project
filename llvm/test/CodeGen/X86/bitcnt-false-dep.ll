; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=haswell | FileCheck %s --check-prefix=HSW
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake | FileCheck %s --check-prefix=SKL
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skx | FileCheck %s --check-prefix=SKL
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=alderlake | FileCheck %s --check-prefix=ADL
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=silvermont -mattr=+lzcnt,+bmi | FileCheck %s --check-prefix=SKL
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=goldmont -mattr=+lzcnt,+bmi | FileCheck %s --check-prefix=SKL
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=znver4 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=znver5 | FileCheck %s --check-prefix=ZEN5
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=znver6 | FileCheck %s --check-prefix=FAST

; This tests a fix for bugzilla 33869 https://bugs.llvm.org/show_bug.cgi?id=33869

declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctpop.i64(i64)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)

define i32 @loopdep_popcnt32(ptr nocapture %x, ptr nocapture %y) nounwind {
entry:
  %vx = load i32, ptr %x
  br label %loop
loop:
  %i = phi i32 [ 1, %entry ], [ %inc, %loop ]
  %s1 = phi i32 [ %vx, %entry ], [ %s2, %loop ]
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %j = tail call i32 @llvm.ctpop.i32(i32 %i)
  %s2 = add i32 %s1, %j
  %inc = add nsw i32 %i, 1
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %exitcond = icmp eq i32 %inc, 156250000
  br i1 %exitcond, label %ret, label %loop
ret:
  ret i32 %s2

;HSW-LABEL:@loopdep_popcnt32
;HSW: xorl [[GPR0:%e[a-d]x]], [[GPR0]]
;HSW-NEXT: popcntl {{.*}}, [[GPR0]]

;SKL-LABEL:@loopdep_popcnt32
;SKL: xorl [[GPR0:%e[a-d]x]], [[GPR0]]
;SKL-NEXT: popcntl {{.*}}, [[GPR0]]

;ADL-LABEL:@loopdep_popcnt32
;ADL-NOT: xor
;ADL: popcntl

;ZEN5-LABEL:@loopdep_popcnt32
;ZEN5-NOT: xor
;ZEN5: popcntl

;FAST-LABEL:@loopdep_popcnt32
;FAST-NOT: xor
;FAST: popcntl
}

define i64 @loopdep_popcnt64(ptr nocapture %x, ptr nocapture %y) nounwind {
entry:
  %vx = load i64, ptr %x
  br label %loop
loop:
  %i = phi i64 [ 1, %entry ], [ %inc, %loop ]
  %s1 = phi i64 [ %vx, %entry ], [ %s2, %loop ]
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %j = tail call i64 @llvm.ctpop.i64(i64 %i)
  %s2 = add i64 %s1, %j
  %inc = add nsw i64 %i, 1
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %exitcond = icmp eq i64 %inc, 156250000
  br i1 %exitcond, label %ret, label %loop
ret:
  ret i64 %s2

;HSW-LABEL:@loopdep_popcnt64
;HSW: xorl %e[[GPR0:[a-d]x]], %e[[GPR0]]
;HSW-NEXT: popcntq {{.*}}, %r[[GPR0]]

;SKL-LABEL:@loopdep_popcnt64
;SKL: xorl %e[[GPR0:[a-d]x]], %e[[GPR0]]
;SKL-NEXT: popcntq {{.*}}, %r[[GPR0]]

;ADL-LABEL:@loopdep_popcnt64
;ADL-NOT: xor
;ADL: popcntq

;ZEN5-LABEL:@loopdep_popcnt64
;ZEN5-NOT: xor
;ZEN5: popcntq

;FAST-LABEL:@loopdep_popcnt64
;FAST-NOT: xor
;FAST: popcntq
}

define i32 @loopdep_tzct32(ptr nocapture %x, ptr nocapture %y) nounwind {
entry:
  %vx = load i32, ptr %x
  br label %loop
loop:
  %i = phi i32 [ 1, %entry ], [ %inc, %loop ]
  %s1 = phi i32 [ %vx, %entry ], [ %s2, %loop ]
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %j = call i32 @llvm.cttz.i32(i32 %i, i1 true)
  %s2 = add i32 %s1, %j
  %inc = add nsw i32 %i, 1
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %exitcond = icmp eq i32 %inc, 156250000
  br i1 %exitcond, label %ret, label %loop
ret:
  ret i32 %s2

;HSW-LABEL:@loopdep_tzct32
;HSW: xorl [[GPR0:%e[a-d]x]], [[GPR0]]
;HSW-NEXT: tzcntl {{.*}}, [[GPR0]]

; This false dependecy issue was fixed in Skylake
;SKL-LABEL:@loopdep_tzct32
;SKL-NOT: xor
;SKL: tzcntl

;ZEN5-LABEL:@loopdep_tzct32
;ZEN5: xorl [[GPR0:%e[a-d]x]], [[GPR0]]
;ZEN5-NEXT: tzcntl {{.*}}, [[GPR0]]

;FAST-LABEL:@loopdep_tzct32
;FAST-NOT: xor
;FAST: tzcntl
}

define i64 @loopdep_tzct64(ptr nocapture %x, ptr nocapture %y) nounwind {
entry:
  %vx = load i64, ptr %x
  br label %loop
loop:
  %i = phi i64 [ 1, %entry ], [ %inc, %loop ]
  %s1 = phi i64 [ %vx, %entry ], [ %s2, %loop ]
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %j = tail call i64 @llvm.cttz.i64(i64 %i, i1 true)
  %s2 = add i64 %s1, %j
  %inc = add nsw i64 %i, 1
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %exitcond = icmp eq i64 %inc, 156250000
  br i1 %exitcond, label %ret, label %loop
ret:
  ret i64 %s2

;HSW-LABEL:@loopdep_tzct64
;HSW: xorl %e[[GPR0:[a-d]x]], %e[[GPR0]]
;HSW-NEXT: tzcntq {{.*}}, %r[[GPR0]]

; This false dependecy issue was fixed in Skylake
;SKL-LABEL:@loopdep_tzct64
;SKL-NOT: xor
;SKL: tzcntq

;ZEN5-LABEL:@loopdep_tzct64
;ZEN5: xorl %e[[GPR0:[a-d]x]], %e[[GPR0]]
;ZEN5-NEXT: tzcntq {{.*}}, %r[[GPR0]]

;FAST-LABEL:@loopdep_tzct64
;FAST-NOT: xor
;FAST: tzcntq
}

define i32 @loopdep_lzct32(ptr nocapture %x, ptr nocapture %y) nounwind {
entry:
  %vx = load i32, ptr %x
  br label %loop
loop:
  %i = phi i32 [ 1, %entry ], [ %inc, %loop ]
  %s1 = phi i32 [ %vx, %entry ], [ %s2, %loop ]
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %j = call i32 @llvm.ctlz.i32(i32 %i, i1 true)
  %s2 = add i32 %s1, %j
  %inc = add nsw i32 %i, 1
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %exitcond = icmp eq i32 %inc, 156250000
  br i1 %exitcond, label %ret, label %loop
ret:
  ret i32 %s2

;HSW-LABEL:@loopdep_lzct32
;HSW: xorl [[GPR0:%e[a-d]x]], [[GPR0]]
;HSW-NEXT: lzcntl {{.*}}, [[GPR0]]

; This false dependecy issue was fixed in Skylake
;SKL-LABEL:@loopdep_lzct32
;SKL-NOT: xor
;SKL: lzcntl

;ZEN5-LABEL:@loopdep_lzct32
;ZEN5-NOT: xor
;ZEN5: lzcntl

;FAST-LABEL:@loopdep_lzct32
;FAST-NOT: xor
;FAST: lzcntl
}

define i64 @loopdep_lzct64(ptr nocapture %x, ptr nocapture %y) nounwind {
entry:
  %vx = load i64, ptr %x
  br label %loop
loop:
  %i = phi i64 [ 1, %entry ], [ %inc, %loop ]
  %s1 = phi i64 [ %vx, %entry ], [ %s2, %loop ]
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %j = tail call i64 @llvm.ctlz.i64(i64 %i, i1 true)
  %s2 = add i64 %s1, %j
  %inc = add nsw i64 %i, 1
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %exitcond = icmp eq i64 %inc, 156250000
  br i1 %exitcond, label %ret, label %loop
ret:
  ret i64 %s2

;HSW-LABEL:@loopdep_lzct64
;HSW: xorl %e[[GPR0:[a-d]x]], %e[[GPR0]]
;HSW-NEXT: lzcntq {{.*}}, %r[[GPR0]]

; This false dependecy issue was fixed in Skylake
;SKL-LABEL:@loopdep_lzct64
;SKL-NOT: xor
;SKL: lzcntq

;ZEN5-LABEL:@loopdep_lzct64
;ZEN5-NOT: xor
;ZEN5: lzcntq

;FAST-LABEL:@loopdep_lzct64
;FAST-NOT: xor
;FAST: lzcntq
}
