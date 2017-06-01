; RUN: opt -indvars -S %s | FileCheck %s

%struct.A = type { i8 }

@c = global %struct.A* null
@d = global i32 4

define void @_Z3fn1v() {
  %x2 = load i32, i32* @d
  %x3 = icmp slt i32 %x2, 1
  %x4 = select i1 %x3, i32 1, i32 %x2
  %x5 = load %struct.A*, %struct.A** @c
  %j.sroa.0.0..sroa_idx = getelementptr %struct.A, %struct.A* %x5, i64 0, i32 0
  %j.sroa.0.0.copyload = load i8, i8* %j.sroa.0.0..sroa_idx
  br label %.preheader4.lr.ph

.preheader4.lr.ph:                                ; preds = %0
  ; CHECK-NOT: add i64 {{.*}}, 4294967296
  br label %.preheader4

.preheader4:                                      ; preds = %x22, %.preheader4.lr.ph
  %k.09 = phi i8* [ undef, %.preheader4.lr.ph ], [ %x25, %x22 ]
  %x8 = icmp ult i32 0, 4
  br i1 %x8, label %.preheader.lr.ph, label %x22

.preheader.lr.ph:                                 ; preds = %.preheader4
  br label %.preheader

.preheader:                                       ; preds = %x17, %.preheader.lr.ph
  %k.17 = phi i8* [ %k.09, %.preheader.lr.ph ], [ %x19, %x17 ]
  %v.06 = phi i32 [ 0, %.preheader.lr.ph ], [ %x20, %x17 ]
  br label %x17

x17:                                              ; preds = %.preheader
  %x18 = sext i8 %j.sroa.0.0.copyload to i64
  %x19 = getelementptr i8, i8* %k.17, i64 %x18
  %x20 = add i32 %v.06, 1
  %x21 = icmp ult i32 %x20, %x4
  br i1 %x21, label %.preheader, label %._crit_edge.8

._crit_edge.8:                                    ; preds = %x17
  %split = phi i8* [ %x19, %x17 ]
  br label %x22

x22:                                              ; preds = %._crit_edge.8, %.preheader4
  %k.1.lcssa = phi i8* [ %split, %._crit_edge.8 ], [ %k.09, %.preheader4 ]
  %x25 = getelementptr i8, i8* %k.1.lcssa
  br label %.preheader4
}
