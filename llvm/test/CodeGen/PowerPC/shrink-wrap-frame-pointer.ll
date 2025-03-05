; Test file to check shrink-wrap pass

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr9 | FileCheck %s --check-prefixes=POWERPC32-AIX
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr9 | FileCheck %s --check-prefixes=POWERPC64-AIX

@.str = private unnamed_addr constant [50 x i8] c"parent_frame_pointer > __builtin_frame_address(0)\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"bad.cpp\00", align 1

; Function Attrs: mustprogress noinline nounwind
define void @_Z3fooPv(ptr noundef readnone %parent_frame_pointer) local_unnamed_addr #0 {

; POWERPC32-AIX-LABEL: ._Z3fooPv:
; POWERPC32-AIX:       # %bb.0:
; POWERPC32-AIX-NEXT:  mflr 0
; POWERPC32-AIX-NEXT:  stwu 1, -64(1)
; POWERPC32-AIX-NEXT:  cmplw 3, 1
; POWERPC32-AIX-NEXT:  stw 0, 72(1)
; POWERPC32-AIX-NEXT:  ble- 0, L..BB0_2
; POWERPC32-AIX-NEXT:  # %bb.1:
; POWERPC32-AIX-NEXT:  addi 1, 1, 64
; POWERPC32-AIX-NEXT:  lwz 0, 8(1)
; POWERPC32-AIX-NEXT:  mtlr 0
; POWERPC32-AIX-NEXT:  blr
; POWERPC32-AIX-NEXT: L..BB0_2:
; POWERPC32-AIX-NEXT:  lwz 4, L..C0(2)
; POWERPC32-AIX-NEXT:  li 5, 6
; POWERPC32-AIX-NEXT:  addi 3, 4, 8
; POWERPC32-AIX-NEXT:  bl .__assert[PR]
; POWERPC32-AIX-NEXT:  nop

; POWERPC64-AIX-LABEL: ._Z3fooPv:
; POWERPC64-AIX:       # %bb.0:
; POWERPC64-AIX-NEXT:  mflr 0
; POWERPC64-AIX-NEXT:  stdu 1, -112(1)
; POWERPC64-AIX-NEXT:  cmpld 3, 1
; POWERPC64-AIX-NEXT:  std 0, 128(1)
; POWERPC64-AIX-NEXT:  ble- 0, L..BB0_2
; POWERPC64-AIX-NEXT:  # %bb.1:
; POWERPC64-AIX-NEXT:  addi 1, 1, 112
; POWERPC64-AIX-NEXT:  ld 0, 16(1)
; POWERPC64-AIX-NEXT:  mtlr 0
; POWERPC64-AIX-NEXT:  blr
; POWERPC64-AIX-NEXT: L..BB0_2:
; POWERPC64-AIX-NEXT:  ld 4, L..C0(2)
; POWERPC64-AIX-NEXT:  li 5, 6
; POWERPC64-AIX-NEXT:  addi 3, 4, 8
; POWERPC64-AIX-NEXT:  bl .__assert[PR]
; POWERPC64-AIX-NEXT:  nop

entry:
  %0 = tail call ptr @llvm.frameaddress.p0(i32 0)
  %cmp = icmp ugt ptr %parent_frame_pointer, %0
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  tail call void @__assert(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 6) #4
  unreachable

cond.end:                                         ; preds = %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.frameaddress.p0(i32 immarg) #1

; Function Attrs: noreturn nounwind
declare void @__assert(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress norecurse nounwind
define noundef i32 @main() local_unnamed_addr #3 {
; POWERPC32-AIX-LABEL: .main:
; POWERPC32-AIX:       # %bb.0:
; POWERPC32-AIX-NEXT:  mflr 0
; POWERPC32-AIX-NEXT:  stwu 1, -64(1)
; POWERPC32-AIX-NEXT:  mr 3, 1
; POWERPC32-AIX-NEXT:  stw 0, 72(1)
; POWERPC32-AIX-NEXT:  bl ._Z3fooPv
; POWERPC32-AIX-NEXT:  nop
; POWERPC32-AIX-NEXT:  li 3, 0
; POWERPC32-AIX-NEXT:  addi 1, 1, 64
; POWERPC32-AIX-NEXT:  lwz 0, 8(1)
; POWERPC32-AIX-NEXT:  mtlr 0
; POWERPC32-AIX-NEXT:  blr

; POWERPC64-AIX-LABEL: .main:
; POWERPC64-AIX:       # %bb.0:
; POWERPC64-AIX-NEXT:  mflr 0
; POWERPC64-AIX-NEXT:  stdu 1, -112(1)
; POWERPC64-AIX-NEXT:  mr 3, 1
; POWERPC64-AIX-NEXT:  std 0, 128(1)
; POWERPC64-AIX-NEXT:  bl ._Z3fooPv
; POWERPC64-AIX-NEXT:  nop
; POWERPC64-AIX-NEXT:  li 3, 0
; POWERPC64-AIX-NEXT:  addi 1, 1, 112
; POWERPC64-AIX-NEXT:  ld 0, 16(1)
; POWERPC64-AIX-NEXT:  mtlr 0
; POWERPC64-AIX-NEXT:  blr

entry:
  %0 = tail call ptr @llvm.frameaddress.p0(i32 0)
  tail call void @_Z3fooPv(ptr noundef %0)
  ret i32 0
}
