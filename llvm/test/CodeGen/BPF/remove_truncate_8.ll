; RUN: llc < %s -mtriple=bpf -mcpu=v1 -verify-machineinstrs | FileCheck %s
; Source Code:
;   struct loc_prog {
;     unsigned int ip;
;     int len;
;   };
;   int exec_prog(struct loc_prog *prog) {
;     if (prog->ip < prog->len) {
;       int x = prog->ip;
;       if (x < 3)
;         prog->ip += 2;
;     }
;     return 3;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm t.c

%struct.loc_prog = type { i32, i32 }

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local i32 @exec_prog(ptr nocapture %prog) local_unnamed_addr {
entry:
  %0 = load i32, ptr %prog, align 4
  %len = getelementptr inbounds %struct.loc_prog, ptr %prog, i64 0, i32 1
  %1 = load i32, ptr %len, align 4
  %cmp = icmp ult i32 %0, %1
  %cmp2 = icmp slt i32 %0, 3
  %or.cond = and i1 %cmp2, %cmp
; CHECK: r{{[0-9]+}} <<= 32
; CHECK: r{{[0-9]+}} s>>= 32
  br i1 %or.cond, label %if.then3, label %if.end5

if.then3:                                         ; preds = %entry
  %add = add nsw i32 %0, 2
  store i32 %add, ptr %prog, align 4
  br label %if.end5

if.end5:                                          ; preds = %if.then3, %entry
  ret i32 3
}
