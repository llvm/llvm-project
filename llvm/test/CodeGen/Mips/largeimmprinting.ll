; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefix=32
; RUN: llc -march=mips64el -mcpu=mips4 -target-abi=n64 -relocation-model=pic < %s | \
; RUN:     FileCheck %s -check-prefix=64
; RUN: llc -march=mips64el -mcpu=mips64 -target-abi=n64 -relocation-model=pic < %s | \
; RUN:     FileCheck %s -check-prefix=64

%struct.S1 = type { [65536 x i8] }

@s1 = external global %struct.S1

define void @f() nounwind {
entry:
; 32:  lui     $[[R0:[0-9]+]], 1
; 32:  addiu   $[[R0]], $[[R0]], 24
; 32:  subu    $sp, $sp, $[[R0]]
; 32:  lui     $[[R1:[0-9]+]], 1
; 32:  addu    $[[R1]], $sp, $[[R1]]
; 32:  sw      $ra, 20($[[R1]])

; 64:  lui     $[[R0:[0-9]+]], 1
; 64:  daddiu  $[[R0]], $[[R0]], 32
; 64:  dsubu   $sp, $sp, $[[R0]]
; 64:  lui     $[[R1:[0-9]+]], 1
; 64:  daddu   $[[R1]], $sp, $[[R1]]
; 64:  sd      $ra, 24($[[R1]])

  %agg.tmp = alloca %struct.S1, align 8
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %agg.tmp, ptr align 1 @s1, i32 65536, i1 false)
  call void @f2(ptr byval(%struct.S1) %agg.tmp) nounwind
  ret void
}

declare void @f2(ptr byval(%struct.S1))

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
