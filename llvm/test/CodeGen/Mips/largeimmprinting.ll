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

  %agg.tmp = alloca %struct.S1, align 1
  %tmp = getelementptr inbounds %struct.S1, %struct.S1* %agg.tmp, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* getelementptr inbounds (%struct.S1, %struct.S1* @s1, i32 0, i32 0, i32 0), i32 65536, i32 1, i1 false)
  call void @f2(%struct.S1* byval %agg.tmp) nounwind
  ret void
}

declare void @f2(%struct.S1* byval)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
