; RUN: llc < %s -march=bpf -mattr=+alu32 -verify-machineinstrs | FileCheck %s
;
; Source:
;  struct __sk_buff {
;    unsigned data;
;    unsigned data_end;
;  };
;
;  void * test(int flag, struct __sk_buff *skb)
;  {
;    void *p;
;
;    if (flag) {
;      p = (void *)(long)skb->data;
;      __asm__ __volatile__("": : :"memory");
;    } else {
;      p = (void *)(long)skb->data_end;
;      __asm__ __volatile__("": : :"memory");
;    }
;
;    return p;
;  }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm t.c

%struct.__sk_buff = type { i32, i32 }

define dso_local i8* @test(i32 %flag, %struct.__sk_buff* nocapture readonly %skb) local_unnamed_addr {
entry:
  %tobool = icmp eq i32 %flag, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  %data = getelementptr inbounds %struct.__sk_buff, %struct.__sk_buff* %skb, i64 0, i32 0
  %0 = load i32, i32* %data, align 4
  tail call void asm sideeffect "", "~{memory}"()
  br label %if.end

if.else:
  %data_end = getelementptr inbounds %struct.__sk_buff, %struct.__sk_buff* %skb, i64 0, i32 1
  %1 = load i32, i32* %data_end, align 4
  tail call void asm sideeffect "", "~{memory}"()
  br label %if.end

if.end:
  %p.0.in.in = phi i32 [ %0, %if.then ], [ %1, %if.else ]
  %p.0.in = zext i32 %p.0.in.in to i64
  %p.0 = inttoptr i64 %p.0.in to i8*
  ret i8* %p.0
}

; CHECK:       w0 = *(u32 *)(r2 + 0)
; CHECK:       w0 = *(u32 *)(r2 + 4)
; CHECK-NOT:   r[[#]] = w[[#]]
; CHECK:       exit
