; RUN: opt -O2 -mtriple=bpf-pc-linux -S < %s | llc -mtriple=bpfel -mcpu=v3 | FileCheck %s
;
; The verifier checks every argument of a global function against its BTF
; type, whether the function uses the argument or not. @foo does not use
; %skb, but the caller must still pass its context in r1, not zero.
;
; Generated from:
;   __noinline int foo(struct __sk_buff *skb)
;   {
;     return bpf_get_prandom_u32();
;   }
;
;   SEC("cgroup_skb/ingress")
;   int prog(struct __sk_buff *skb)
;   {
;     return foo(skb);
;   }

target triple = "bpf"

define dso_local i32 @foo(ptr noundef %skb) noinline {
  %r = call i32 inttoptr (i64 7 to ptr)()
  ret i32 %r
}

define dso_local i32 @prog(ptr noundef %skb) section "cgroup_skb/ingress" {
; CHECK-LABEL: prog:
; CHECK-NOT:   r1 =
; CHECK:       call foo
  %r = call i32 @foo(ptr noundef %skb)
  ret i32 %r
}
