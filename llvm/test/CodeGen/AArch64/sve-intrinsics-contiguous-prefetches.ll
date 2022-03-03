; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+streaming-sve < %s | FileCheck %s

;
; Testing prfop encodings
;
define void @test_svprf_pldl1strm(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pldl1strm
; CHECK: prfb pldl1strm, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 1)
  ret void
}

define void @test_svprf_pldl2keep(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pldl2keep
; CHECK: prfb pldl2keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 2)
  ret void
}

define void @test_svprf_pldl2strm(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pldl2strm
; CHECK: prfb pldl2strm, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 3)
  ret void
}

define void @test_svprf_pldl3keep(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pldl3keep
; CHECK: prfb pldl3keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 4)
  ret void
}

define void @test_svprf_pldl3strm(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pldl3strm
; CHECK: prfb pldl3strm, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 5)
  ret void
}

define void @test_svprf_pstl1keep(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pstl1keep
; CHECK: prfb pstl1keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 8)
  ret void
}

define void @test_svprf_pstl1strm(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pstl1strm
; CHECK: prfb pstl1strm, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 9)
  ret void
}

define void @test_svprf_pstl2keep(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pstl2keep
; CHECK: prfb pstl2keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 10)
  ret void
}

define void @test_svprf_pstl2strm(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pstl2strm
; CHECK: prfb pstl2strm, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 11)
  ret void
}

define void @test_svprf_pstl3keep(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pstl3keep
; CHECK: prfb pstl3keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 12)
  ret void
}

define void @test_svprf_pstl3strm(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprf_pstl3strm
; CHECK: prfb pstl3strm, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 13)
  ret void
}

;
; Testing imm limits of SI form
;

define void @test_svprf_vnum_under(<vscale x 16 x i1> %pg, <vscale x 16 x i8>* %base) {
; CHECK-LABEL: test_svprf_vnum_under
; CHECK-NOT: prfb pstl3strm, p0, [x0, #-33, mul vl]
entry:
  %gep = getelementptr inbounds <vscale x 16 x i8>, <vscale x 16 x i8>* %base, i64 -33, i64 0
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %gep, i32 13)
  ret void
}

define void @test_svprf_vnum_min(<vscale x 16 x i1> %pg, <vscale x 16 x i8>* %base) {
; CHECK-LABEL: test_svprf_vnum_min
; CHECK: prfb pstl3strm, p0, [x0, #-32, mul vl]
entry:
  %gep = getelementptr inbounds <vscale x 16 x i8>, <vscale x 16 x i8>* %base, i64 -32, i64 0
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %gep, i32 13)
  ret void
}

define void @test_svprf_vnum_over(<vscale x 16 x i1> %pg, <vscale x 16 x i8>* %base) {
; CHECK-LABEL: test_svprf_vnum_over
; CHECK-NOT: prfb pstl3strm, p0, [x0, #32, mul vl]
entry:
  %gep = getelementptr inbounds <vscale x 16 x i8>, <vscale x 16 x i8>* %base, i64 32, i64 0
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %gep, i32 13)
  ret void
}

define void @test_svprf_vnum_max(<vscale x 16 x i1> %pg, <vscale x 16 x i8>* %base) {
; CHECK-LABEL: test_svprf_vnum_max
; CHECK: prfb pstl3strm, p0, [x0, #31, mul vl]
entry:
  %gep = getelementptr inbounds <vscale x 16 x i8>, <vscale x 16 x i8>* %base, i64 31, i64 0
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %gep, i32 13)
  ret void
}

;
; scalar contiguous
;

define void @test_svprfb(<vscale x 16 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprfb
; CHECK: prfb pldl1keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 0)
  ret void
}

define void @test_svprfh(<vscale x 8 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprfh
; CHECK: prfh pldl1keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %pg, i8* %base, i32 0)
  ret void
}

define void @test_svprfw(<vscale x 4 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprfw
; CHECK: prfw pldl1keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %pg, i8* %base, i32 0)
  ret void
}

define void @test_svprfd(<vscale x 2 x i1> %pg, i8* %base) {
; CHECK-LABEL: test_svprfd
; CHECK: prfd pldl1keep, p0, [x0]
entry:
  tail call void @llvm.aarch64.sve.prf.nxv2i1(<vscale x 2 x i1> %pg, i8* %base, i32 0)
  ret void
}

;
; scalar + imm contiguous
;
; imm form of prfb is tested above

define void @test_svprfh_vnum(<vscale x 8 x i1> %pg, <vscale x 8 x i16>* %base) {
; CHECK-LABEL: test_svprfh_vnum
; CHECK: prfh pstl3strm, p0, [x0, #31, mul vl]
entry:
  %gep = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %base, i64 31
  %addr = bitcast <vscale x 8 x i16>* %gep to i8*
  tail call void @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %pg, i8* %addr, i32 13)
  ret void
}

define void @test_svprfw_vnum(<vscale x 4 x i1> %pg, <vscale x 4 x i32>* %base) {
; CHECK-LABEL: test_svprfw_vnum
; CHECK: prfw pstl3strm, p0, [x0, #31, mul vl]
entry:
  %gep = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %base, i64 31
  %addr = bitcast <vscale x 4 x i32>* %gep to i8*
  tail call void @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %pg, i8* %addr, i32 13)
  ret void
}

define void @test_svprfd_vnum(<vscale x 2 x i1> %pg, <vscale x 2 x i64>* %base) {
; CHECK-LABEL: test_svprfd_vnum
; CHECK: prfd pstl3strm, p0, [x0, #31, mul vl]
entry:
  %gep = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %base, i64 31
  %addr = bitcast <vscale x 2 x i64>* %gep to i8*
  tail call void @llvm.aarch64.sve.prf.nxv2i1(<vscale x 2 x i1> %pg, i8* %addr, i32 13)
  ret void
}

;
; scalar + scaled scalar contiguous
;

define void @test_svprfb_ss(<vscale x 16 x i1> %pg, i8* %base, i64 %offset) {
; CHECK-LABEL: test_svprfb_ss
; CHECK: prfb pstl3strm, p0, [x0, x1]
entry:
  %addr = getelementptr i8, i8* %base, i64 %offset
  tail call void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %addr, i32 13)
  ret void
}

define void @test_svprfh_ss(<vscale x 8 x i1> %pg, i16* %base, i64 %offset) {
; CHECK-LABEL: test_svprfh_ss
; CHECK: prfh pstl3strm, p0, [x0, x1, lsl #1]
entry:
  %gep = getelementptr i16, i16* %base, i64 %offset
  %addr = bitcast i16* %gep to i8*
  tail call void @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %pg, i8* %addr, i32 13)
  ret void
}

define void @test_svprfw_ss(<vscale x 4 x i1> %pg, i32* %base, i64 %offset) {
; CHECK-LABEL: test_svprfw_ss
; CHECK: prfw pstl3strm, p0, [x0, x1, lsl #2]
entry:
  %gep = getelementptr i32, i32* %base, i64 %offset
  %addr = bitcast i32* %gep to i8*
  tail call void @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %pg, i8* %addr, i32 13)
  ret void
}

define void @test_svprfd_ss(<vscale x 2 x i1> %pg, i64* %base, i64 %offset) {
; CHECK-LABEL: test_svprfd_ss
; CHECK: prfd pstl3strm, p0, [x0, x1, lsl #3]
entry:
  %gep = getelementptr i64, i64* %base, i64 %offset
  %addr = bitcast i64* %gep to i8*
  tail call void @llvm.aarch64.sve.prf.nxv2i1(<vscale x 2 x i1> %pg, i8* %addr, i32 13)
  ret void
}


declare void @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1>, i8*, i32)
declare void @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1>,  i8*, i32)
declare void @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1>,  i8*, i32)
declare void @llvm.aarch64.sve.prf.nxv2i1(<vscale x 2 x i1>,  i8*, i32)
