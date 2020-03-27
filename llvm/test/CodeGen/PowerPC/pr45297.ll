; RUN: not --crash llc -verify-machineinstrs \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu -mattr=+altivec \
; RUN:   -mattr=-power8-vector -mattr=-vsx < %s 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: Cannot select: t{{[0-9]+}}: ch = PPCISD::ST_VSR_SCAL_INT<(store 4 into @Global)>

@Global = dso_local global i32 55, align 4

define dso_local void @test(float %0) local_unnamed_addr {
entry:
  %1 = fptosi float %0 to i32
  store i32 %1, i32* @Global, align 4
  ret void
}
