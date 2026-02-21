; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -O1 --disable-packetizer -filetype=obj -o /dev/null %s
; REQUIRES: asserts

; Check that compiling with --disable-packetizer does not crash.
; When packetization is disabled, new value jumps should not be generated
; because the producer instruction cannot be in the same packet as the
; consumer (.new) instruction.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-linux-musl"

; This pattern triggers new value jump generation:
define i32 @test_nvj_disable_packetizer() {
entry:
  %ptr = load ptr, ptr null, align 4
  %cmp = icmp eq ptr %ptr, null
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @llvm.memset.p0.i64(ptr null, i8 0, i64 19, i1 false)
  br label %exit

if.else:
  %val = atomicrmw sub ptr null, i32 0 monotonic, align 4
  br label %exit

exit:
  %result = phi i32 [ 0, %if.then ], [ 1, %if.else ]
  ret i32 %result
}

declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
