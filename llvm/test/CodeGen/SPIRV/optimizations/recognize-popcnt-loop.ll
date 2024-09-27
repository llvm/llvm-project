; RUN: opt -O3 -mtriple=spirv32-- %s -o - | llc -O3 -mtriple=spirv32-- -o - | FileCheck %s
; RUN: %if spirv-tools %{ opt -O3 -mtriple=spirv32-- %s -o - | llc -O3 -mtriple=spirv32-- -o - -filetype=obj | spirv-val %}

; RUN: opt -O3 -mtriple=spirv64-- %s -o - | llc -O3 -mtriple=spirv64-- -o - | FileCheck %s
; RUN: %if spirv-tools %{ opt -O3 -mtriple=spirv64-- %s -o - | llc -O3 -mtriple=spirv64-- -o - -filetype=obj | spirv-val %}

; Mostly copied from x86 version.

;To recognize this pattern:
;int popcount(unsigned long long a) {
;    int c = 0;
;    while (a) {
;        c++;
;        a &= a - 1;
;    }
;    return c;
;}

; CHECK-DAG: OpName %[[POPCNT64:.*]] "popcount_i64"
; CHECK-DAG: OpName %[[POPCNT32:.*]] "popcount_i32"
; CHECK-DAG: OpName %[[POPCNT2:.*]] "popcount2"
; CHECK-DAG: %[[INT64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[INT32:.*]] = OpTypeInt 32 0

define i32 @popcount_i64(i64 %a) nounwind uwtable readnone ssp {
entry:
  %tobool3 = icmp eq i64 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i64 %a.addr.04, -1
  %and = and i64 %sub, %a.addr.04
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}
; CHECK-DAG: %[[POPCNT64]] = OpFunction
; CHECK: %[[A:.*]] = OpFunctionParameter %[[INT64]]
; CHECK-DAG: %{{.+}} = OpBitCount %[[INT64]] %[[A]]
; CHECK-DAG: OpFunctionEnd


define i32 @popcount_i32(i32 %a) nounwind uwtable readnone ssp {
entry:
  %tobool3 = icmp eq i32 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i32 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i32 %a.addr.04, -1
  %and = and i32 %sub, %a.addr.04
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}
; CHECK: %[[POPCNT32]] = OpFunction
; CHECK: %[[A:.*]] = OpFunctionParameter %[[INT32]]
; CHECK-DAG: %{{.*}} = OpBitCount %[[INT32]] %[[A]]
; CHECK-DAG: OpFunctionEnd

; To recognize this pattern:
;int popcount(unsigned long long a, int mydata1, int mydata2) {
;    int c = 0;
;    while (a) {
;        c++;
;        a &= a - 1;
;        mydata1 *= c;
;        mydata2 *= (int)a;
;    }
;    return c + mydata1 + mydata2;
;}

define i32 @popcount2(i64 %a, i32 %mydata1, i32 %mydata2) nounwind uwtable readnone ssp {
entry:
  %tobool9 = icmp eq i64 %a, 0
  br i1 %tobool9, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.013 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %mydata2.addr.012 = phi i32 [ %mul1, %while.body ], [ %mydata2, %entry ]
  %mydata1.addr.011 = phi i32 [ %mul, %while.body ], [ %mydata1, %entry ]
  %a.addr.010 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.013, 1
  %sub = add i64 %a.addr.010, -1
  %and = and i64 %sub, %a.addr.010
  %mul = mul nsw i32 %inc, %mydata1.addr.011
  %conv = trunc i64 %and to i32
  %mul1 = mul nsw i32 %conv, %mydata2.addr.012
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %mydata2.addr.0.lcssa = phi i32 [ %mydata2, %entry ], [ %mul1, %while.body ]
  %mydata1.addr.0.lcssa = phi i32 [ %mydata1, %entry ], [ %mul, %while.body ]
  %add = add i32 %mydata2.addr.0.lcssa, %mydata1.addr.0.lcssa
  %add2 = add i32 %add, %c.0.lcssa
  ret i32 %add2
}
; CHECK: %[[POPCNT2]] = OpFunction
; CHECK: %[[A:.*]] = OpFunctionParameter %[[INT64]]
; CHECK-DAG: %{{.*}} = OpBitCount %[[INT64]] %[[A]]
; CHECK-DAG: OpFunctionEnd
