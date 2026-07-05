; RUN: llc < %s -mtriple=bpf -verify-machineinstrs | FileCheck %s

; Source code:
;struct __sk_buff;
;unsigned long long
;load_byte(ptr skb, unsigned long long off) asm("llvm.bpf.load.byte");
;unsigned long long
;load_half(ptr skb, unsigned long long off) asm("llvm.bpf.load.half");
;typedef unsigned char      uint8_t;
;typedef unsigned short     uint16_t;
;
;int func_b(struct __sk_buff *skb)
;{
;    uint8_t t = load_byte(skb, 0);
;    return t;
;}
;
;int func_h(struct __sk_buff *skb)
;{
;    uint16_t t = load_half(skb, 0);
;    return t;
;}
;
;int func_w(struct __sk_buff *skb)
;{
;    uint32_t t = load_word(skb, 0);
;    return t;
;}

%struct.__sk_buff = type opaque

; Function Attrs: nounwind readonly
define i32 @func_b(ptr %skb) local_unnamed_addr #0 {
entry:
  %call = tail call i64 @llvm.bpf.load.byte(ptr %skb, i64 0)
  %conv = trunc i64 %call to i32
  %conv1 = and i32 %conv, 255
; CHECK-NOT:  r0 &= 255
  ret i32 %conv1
}

; Function Attrs: nounwind readonly
declare i64 @llvm.bpf.load.byte(ptr, i64) #1

; Function Attrs: nounwind readonly
define i32 @func_h(ptr %skb) local_unnamed_addr #0 {
entry:
  %call = tail call i64 @llvm.bpf.load.half(ptr %skb, i64 0)
  %conv = trunc i64 %call to i32
  %conv1 = and i32 %conv, 65535
; CHECK-NOT:  r0 &= 65535
  ret i32 %conv1
}

; Function Attrs: nounwind readonly
declare i64 @llvm.bpf.load.half(ptr, i64) #1
