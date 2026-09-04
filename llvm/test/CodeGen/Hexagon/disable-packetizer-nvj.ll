; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -O2 --disable-packetizer -filetype=obj -o /dev/null %s
; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -O2 -mattr=-nvj -filetype=obj -o /dev/null %s
; REQUIRES: asserts

; Check that compiling with --disable-packetizer does not crash.
; New value jumps require the feeder instruction to be in the same packet
; as the consumer (.new) instruction. When packetization is disabled, the
; NVJ pass must be skipped to avoid an assertion in the MCCodeEmitter
; ("Couldn't find producer").
;
; The branches here are non-if-convertible (due to calls), ensuring the
; compare+branch pattern survives to the NVJ pass regardless of the
; optimization pipeline used.

declare void @do_work(i32)
declare void @do_other(i32)

define void @test_nvj(ptr %p, i32 %threshold) {
entry:
  %val = load i32, ptr %p, align 4
  %cmp = icmp eq i32 %val, %threshold
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @do_work(i32 %val)
  br label %exit

if.else:
  call void @do_other(i32 %val)
  br label %exit

exit:
  ret void
}
