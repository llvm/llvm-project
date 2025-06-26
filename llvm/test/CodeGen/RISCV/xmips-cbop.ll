; RUN: llc -mtriple=riscv32 -mattr=+xmipscbop -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32XMIPSPREFETCH
; RUN: llc -mtriple=riscv64 -mattr=+xmipscbop -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64XMIPSPREFETCH

define void @prefetch_read(ptr noundef %ptr) nounwind {
; RV32XMIPSPREFETCH-LABEL: prefetch_read:
; RV32XMIPSPREFETCH:    mips.pref       8, 1(a0)
;
; RV64XMIPSPREFETCH-LABEL: prefetch_read:
; RV64XMIPSPREFETCH:    mips.pref       8, 1(a0)
entry:
  %arrayidx = getelementptr inbounds nuw i8, ptr %ptr, i64 1
  tail call void @llvm.prefetch.p0(ptr nonnull %arrayidx, i32 0, i32 0, i32 1)
  ret void
  ret void
}
 
define void @prefetch_write(ptr noundef %ptr) nounwind  {
; RV32XMIPSPREFETCH-LABEL: prefetch_write:
; RV32XMIPSPREFETCH:         addi    a0, a0, 512  
; RV32XMIPSPREFETCH-NEXT:    mips.pref       9, 0(a0)
;
; RV64XMIPSPREFETCH-LABEL: prefetch_write:
; RV64XMIPSPREFETCH:         addi    a0, a0, 512 
; RV64XMIPSPREFETCH-NEXT:    mips.pref       9, 0(a0)
entry:
  %arrayidx = getelementptr inbounds nuw i8, ptr %ptr, i64 512
  tail call void @llvm.prefetch.p0(ptr nonnull %arrayidx, i32 1, i32 0, i32 1)
  ret void
}

