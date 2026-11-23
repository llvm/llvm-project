; RUN: llc -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv68 \
; RUN:     -relocation-model=pic -hexagon-small-data-threshold=0 \
; RUN:     -verify-machineinstrs < %s -o /dev/null

; Check HexagonOptAddrMode does not leave a use of a killed register when
; chaining A2_addi base computations (PR240245).

target triple = "hexagon-unknown-linux-musl"

@g = internal global [139200 x i8] zeroinitializer, align 64

define hidden void @force_unlock() local_unnamed_addr {
entry:
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 139168) release, align 32
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 138912) release, align 32
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 7488) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 7424) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 7360) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 7296) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 7232) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 7168) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 7104) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 7040) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6976) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6912) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6848) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6784) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6720) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6656) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6592) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6528) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6464) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6400) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6336) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6272) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6208) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6144) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6080) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 6016) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5952) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5888) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5824) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5760) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5696) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5632) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5568) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5504) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5440) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5376) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5312) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5248) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5184) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5120) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 5056) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4992) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4928) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4864) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4800) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4736) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4672) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4608) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4544) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4480) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4416) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4352) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4288) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4224) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4160) release, align 64
  store atomic volatile i8 0, ptr getelementptr inbounds nuw (i8, ptr @g, i32 4096) release, align 64
  ret void
}
