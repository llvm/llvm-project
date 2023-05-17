; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 -disable-lsr -verify-machineinstrs -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOSTRICTALIGN %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+strict-align -aarch64-enable-atomic-cfg-tidy=0 -disable-lsr -verify-machineinstrs -o - %s | FileCheck --check-prefix=CHECK --check-prefix=STRICTALIGN %s

; This file contains tests for the AArch64 load/store optimizer.

%padding = type { ptr, ptr, ptr, ptr }
%s.byte = type { i8, i8 }
%s.halfword = type { i16, i16 }
%s.word = type { i32, i32 }
%s.doubleword = type { i64, i32 }
%s.quadword = type { fp128, i32 }
%s.float = type { float, i32 }
%s.double = type { double, i32 }
%struct.byte = type { %padding, %s.byte }
%struct.halfword = type { %padding, %s.halfword }
%struct.word = type { %padding, %s.word }
%struct.doubleword = type { %padding, %s.doubleword }
%struct.quadword = type { %padding, %s.quadword }
%struct.float = type { %padding, %s.float }
%struct.double = type { %padding, %s.double }

; Check the following transform:
;
; (ldr|str) X, [x0, #32]
;  ...
; add x0, x0, #32
;  ->
; (ldr|str) X, [x0, #32]!
;
; with X being either w1, x1, s0, d0 or q0.

declare void @bar_byte(ptr, i8)

define void @load-pre-indexed-byte(ptr %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-byte
; CHECK: ldrb w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.byte, ptr %ptr, i64 0, i32 1, i32 0
  %add = load i8, ptr %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.byte, ptr %ptr, i64 0, i32 1
  tail call void @bar_byte(ptr %c, i8 %add)
  ret void
}

define void @store-pre-indexed-byte(ptr %ptr, i8 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-byte
; CHECK: strb w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.byte, ptr %ptr, i64 0, i32 1, i32 0
  store i8 %val, ptr %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.byte, ptr %ptr, i64 0, i32 1
  tail call void @bar_byte(ptr %c, i8 %val)
  ret void
}

declare void @bar_halfword(ptr, i16)

define void @load-pre-indexed-halfword(ptr %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-halfword
; CHECK: ldrh w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.halfword, ptr %ptr, i64 0, i32 1, i32 0
  %add = load i16, ptr %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.halfword, ptr %ptr, i64 0, i32 1
  tail call void @bar_halfword(ptr %c, i16 %add)
  ret void
}

define void @store-pre-indexed-halfword(ptr %ptr, i16 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-halfword
; CHECK: strh w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.halfword, ptr %ptr, i64 0, i32 1, i32 0
  store i16 %val, ptr %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.halfword, ptr %ptr, i64 0, i32 1
  tail call void @bar_halfword(ptr %c, i16 %val)
  ret void
}

declare void @bar_word(ptr, i32)

define void @load-pre-indexed-word(ptr %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-word
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1, i32 0
  %add = load i32, ptr %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1
  tail call void @bar_word(ptr %c, i32 %add)
  ret void
}

define void @store-pre-indexed-word(ptr %ptr, i32 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-word
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1, i32 0
  store i32 %val, ptr %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1
  tail call void @bar_word(ptr %c, i32 %val)
  ret void
}

declare void @bar_doubleword(ptr, i64)

define void @load-pre-indexed-doubleword(ptr %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-doubleword
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.doubleword, ptr %ptr, i64 0, i32 1, i32 0
  %add = load i64, ptr %a, align 8
  br label %bar
bar:
  %c = getelementptr inbounds %struct.doubleword, ptr %ptr, i64 0, i32 1
  tail call void @bar_doubleword(ptr %c, i64 %add)
  ret void
}

define void @store-pre-indexed-doubleword(ptr %ptr, i64 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-doubleword
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.doubleword, ptr %ptr, i64 0, i32 1, i32 0
  store i64 %val, ptr %a, align 8
  br label %bar
bar:
  %c = getelementptr inbounds %struct.doubleword, ptr %ptr, i64 0, i32 1
  tail call void @bar_doubleword(ptr %c, i64 %val)
  ret void
}

declare void @bar_quadword(ptr, fp128)

define void @load-pre-indexed-quadword(ptr %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-quadword
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.quadword, ptr %ptr, i64 0, i32 1, i32 0
  %add = load fp128, ptr %a, align 16
  br label %bar
bar:
  %c = getelementptr inbounds %struct.quadword, ptr %ptr, i64 0, i32 1
  tail call void @bar_quadword(ptr %c, fp128 %add)
  ret void
}

define void @store-pre-indexed-quadword(ptr %ptr, fp128 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-quadword
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.quadword, ptr %ptr, i64 0, i32 1, i32 0
  store fp128 %val, ptr %a, align 16
  br label %bar
bar:
  %c = getelementptr inbounds %struct.quadword, ptr %ptr, i64 0, i32 1
  tail call void @bar_quadword(ptr %c, fp128 %val)
  ret void
}

declare void @bar_float(ptr, float)

define void @load-pre-indexed-float(ptr %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-float
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.float, ptr %ptr, i64 0, i32 1, i32 0
  %add = load float, ptr %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.float, ptr %ptr, i64 0, i32 1
  tail call void @bar_float(ptr %c, float %add)
  ret void
}

define void @store-pre-indexed-float(ptr %ptr, float %val) nounwind {
; CHECK-LABEL: store-pre-indexed-float
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.float, ptr %ptr, i64 0, i32 1, i32 0
  store float %val, ptr %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.float, ptr %ptr, i64 0, i32 1
  tail call void @bar_float(ptr %c, float %val)
  ret void
}

declare void @bar_double(ptr, double)

define void @load-pre-indexed-double(ptr %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-double
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.double, ptr %ptr, i64 0, i32 1, i32 0
  %add = load double, ptr %a, align 8
  br label %bar
bar:
  %c = getelementptr inbounds %struct.double, ptr %ptr, i64 0, i32 1
  tail call void @bar_double(ptr %c, double %add)
  ret void
}

define void @store-pre-indexed-double(ptr %ptr, double %val) nounwind {
; CHECK-LABEL: store-pre-indexed-double
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.double, ptr %ptr, i64 0, i32 1, i32 0
  store double %val, ptr %a, align 8
  br label %bar
bar:
  %c = getelementptr inbounds %struct.double, ptr %ptr, i64 0, i32 1
  tail call void @bar_double(ptr %c, double %val)
  ret void
}

; Check the following transform:
;
; (ldp|stp) w1, w2 [x0, #32]
;  ...
; add x0, x0, #32
;  ->
; (ldp|stp) w1, w2, [x0, #32]!
;

define void @load-pair-pre-indexed-word(ptr %ptr) nounwind {
; CHECK-LABEL: load-pair-pre-indexed-word
; CHECK: ldp w{{[0-9]+}}, w{{[0-9]+}}, [x0, #32]!
; CHECK-NOT: add x0, x0, #32
entry:
  %a = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1, i32 0
  %a1 = load i32, ptr %a, align 4
  %b = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1, i32 1
  %b1 = load i32, ptr %b, align 4
  %add = add i32 %a1, %b1
  br label %bar
bar:
  %c = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1
  tail call void @bar_word(ptr %c, i32 %add)
  ret void
}

define void @store-pair-pre-indexed-word(ptr %ptr, i32 %val) nounwind {
; CHECK-LABEL: store-pair-pre-indexed-word
; CHECK: stp w{{[0-9]+}}, w{{[0-9]+}}, [x0, #32]!
; CHECK-NOT: add x0, x0, #32
entry:
  %a = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1, i32 0
  store i32 %val, ptr %a, align 4
  %b = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1, i32 1
  store i32 %val, ptr %b, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.word, ptr %ptr, i64 0, i32 1
  tail call void @bar_word(ptr %c, i32 %val)
  ret void
}

; Check the following transform:
;
; add x8, x8, #16
;  ...
; ldr X, [x8]
;  ->
; ldr X, [x8, #16]!
;
; with X being either w0, x0, s0, d0 or q0.

%pre.struct.i32 = type { i32, i32, i32, i32, i32}
%pre.struct.i64 = type { i32, i64, i64, i64, i64}
%pre.struct.i128 = type { i32, <2 x i64>, <2 x i64>, <2 x i64>}
%pre.struct.float = type { i32, float, float, float}
%pre.struct.double = type { i32, double, double, double}

define i32 @load-pre-indexed-word2(ptr %this, i1 %cond,
                                   ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-word2
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}, #4]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i32, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i32, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load i32, ptr %retptr
  ret i32 %ret
}

define i64 @load-pre-indexed-doubleword2(ptr %this, i1 %cond,
                                         ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-doubleword2
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i64, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i64, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load i64, ptr %retptr
  ret i64 %ret
}

define <2 x i64> @load-pre-indexed-quadword2(ptr %this, i1 %cond,
                                             ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-quadword2
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}, #16]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i128, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i128, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load <2 x i64>, ptr %retptr
  ret <2 x i64> %ret
}

define float @load-pre-indexed-float2(ptr %this, i1 %cond,
                                      ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-float2
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}, #4]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.float, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.float, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load float, ptr %retptr
  ret float %ret
}

define double @load-pre-indexed-double2(ptr %this, i1 %cond,
                                        ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-double2
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.double, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.double, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load double, ptr %retptr
  ret double %ret
}

define i32 @load-pre-indexed-word3(ptr %this, i1 %cond,
                                   ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-word3
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}, #12]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i32, ptr %load1, i64 0, i32 3
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i32, ptr %load2, i64 0, i32 4
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load i32, ptr %retptr
  ret i32 %ret
}

define i64 @load-pre-indexed-doubleword3(ptr %this, i1 %cond,
                                         ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-doubleword3
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, #16]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i64, ptr %load1, i64 0, i32 2
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i64, ptr %load2, i64 0, i32 3
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load i64, ptr %retptr
  ret i64 %ret
}

define <2 x i64> @load-pre-indexed-quadword3(ptr %this, i1 %cond,
                                             ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-quadword3
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}, #32]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i128, ptr %load1, i64 0, i32 2
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i128, ptr %load2, i64 0, i32 3
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load <2 x i64>, ptr %retptr
  ret <2 x i64> %ret
}

define float @load-pre-indexed-float3(ptr %this, i1 %cond,
                                      ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-float3
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.float, ptr %load1, i64 0, i32 2
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.float, ptr %load2, i64 0, i32 3
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load float, ptr %retptr
  ret float %ret
}

define double @load-pre-indexed-double3(ptr %this, i1 %cond,
                                        ptr %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-double3
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}, #16]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.double, ptr %load1, i64 0, i32 2
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.double, ptr %load2, i64 0, i32 3
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load double, ptr %retptr
  ret double %ret
}

; Check the following transform:
;
; add x8, x8, #16
;  ...
; str X, [x8]
;  ->
; str X, [x8, #16]!
;
; with X being either w0, x0, s0, d0 or q0.

define void @store-pre-indexed-word2(ptr %this, i1 %cond,
                                     ptr %load2,
                                     i32 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-word2
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}, #4]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i32, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i32, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store i32 %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-doubleword2(ptr %this, i1 %cond,
                                           ptr %load2,
                                           i64 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-doubleword2
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i64, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i64, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store i64 %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-quadword2(ptr %this, i1 %cond,
                                         ptr %load2,
                                         <2 x i64> %val) nounwind {
; CHECK-LABEL: store-pre-indexed-quadword2
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}, #16]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i128, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i128, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store <2 x i64> %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-float2(ptr %this, i1 %cond,
                                      ptr %load2,
                                      float %val) nounwind {
; CHECK-LABEL: store-pre-indexed-float2
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}, #4]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.float, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.float, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store float %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-double2(ptr %this, i1 %cond,
                                      ptr %load2,
                                      double %val) nounwind {
; CHECK-LABEL: store-pre-indexed-double2
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.double, ptr %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.double, ptr %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store double %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-word3(ptr %this, i1 %cond,
                                     ptr %load2,
                                     i32 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-word3
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}, #12]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i32, ptr %load1, i64 0, i32 3
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i32, ptr %load2, i64 0, i32 4
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store i32 %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-doubleword3(ptr %this, i1 %cond,
                                           ptr %load2,
                                           i64 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-doubleword3
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}, #24]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i64, ptr %load1, i64 0, i32 3
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i64, ptr %load2, i64 0, i32 4
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store i64 %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-quadword3(ptr %this, i1 %cond,
                                         ptr %load2,
                                         <2 x i64> %val) nounwind {
; CHECK-LABEL: store-pre-indexed-quadword3
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}, #32]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.i128, ptr %load1, i64 0, i32 2
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i128, ptr %load2, i64 0, i32 3
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store <2 x i64> %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-float3(ptr %this, i1 %cond,
                                      ptr %load2,
                                      float %val) nounwind {
; CHECK-LABEL: store-pre-indexed-float3
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.float, ptr %load1, i64 0, i32 2
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.float, ptr %load2, i64 0, i32 3
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store float %val, ptr %retptr
  ret void
}

define void @store-pre-indexed-double3(ptr %this, i1 %cond,
                                      ptr %load2,
                                      double %val) nounwind {
; CHECK-LABEL: store-pre-indexed-double3
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}, #16]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load ptr, ptr %this
  %gep1 = getelementptr inbounds %pre.struct.double, ptr %load1, i64 0, i32 2
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.double, ptr %load2, i64 0, i32 3
  br label %return
return:
  %retptr = phi ptr [ %gep1, %if.then ], [ %gep2, %if.end ]
  store double %val, ptr %retptr
  ret void
}

; Check the following transform:
;
; ldr X, [x20]
;  ...
; add x20, x20, #32
;  ->
; ldr X, [x20], #32
;
; with X being either w0, x0, s0, d0 or q0.

define void @load-post-indexed-byte(ptr %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-byte
; CHECK: ldrb w{{[0-9]+}}, [x{{[0-9]+}}], #4
entry:
  %gep1 = getelementptr i8, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i8, ptr %iv2, i64 -1
  %load = load i8, ptr %gep2
  call void @use-byte(i8 %load)
  %load2 = load i8, ptr %iv2
  call void @use-byte(i8 %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i8, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-halfword(ptr %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-halfword
; CHECK: ldrh w{{[0-9]+}}, [x{{[0-9]+}}], #8
entry:
  %gep1 = getelementptr i16, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i16, ptr %iv2, i64 -1
  %load = load i16, ptr %gep2
  call void @use-halfword(i16 %load)
  %load2 = load i16, ptr %iv2
  call void @use-halfword(i16 %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i16, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-word(ptr %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-word
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}], #16
entry:
  %gep1 = getelementptr i32, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i32, ptr %iv2, i64 -1
  %load = load i32, ptr %gep2
  call void @use-word(i32 %load)
  %load2 = load i32, ptr %iv2
  call void @use-word(i32 %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i32, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-doubleword(ptr %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-doubleword
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}], #32
entry:
  %gep1 = getelementptr i64, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i64, ptr %iv2, i64 -1
  %load = load i64, ptr %gep2
  call void @use-doubleword(i64 %load)
  %load2 = load i64, ptr %iv2
  call void @use-doubleword(i64 %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i64, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-quadword(ptr %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-quadword
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}], #64
entry:
  %gep1 = getelementptr <2 x i64>, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr <2 x i64>, ptr %iv2, i64 -1
  %load = load <2 x i64>, ptr %gep2
  call void @use-quadword(<2 x i64> %load)
  %load2 = load <2 x i64>, ptr %iv2
  call void @use-quadword(<2 x i64> %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr <2 x i64>, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-float(ptr %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-float
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}], #16
entry:
  %gep1 = getelementptr float, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr float, ptr %iv2, i64 -1
  %load = load float, ptr %gep2
  call void @use-float(float %load)
  %load2 = load float, ptr %iv2
  call void @use-float(float %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr float, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-double(ptr %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-double
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}], #32
entry:
  %gep1 = getelementptr double, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr double, ptr %iv2, i64 -1
  %load = load double, ptr %gep2
  call void @use-double(double %load)
  %load2 = load double, ptr %iv2
  call void @use-double(double %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr double, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

; Check the following transform:
;
; str X, [x20]
;  ...
; add x20, x20, #32
;  ->
; str X, [x20], #32
;
; with X being either w0, x0, s0, d0 or q0.

define void @store-post-indexed-byte(ptr %array, i64 %count, i8 %val) nounwind {
; CHECK-LABEL: store-post-indexed-byte
; CHECK: strb w{{[0-9]+}}, [x{{[0-9]+}}], #4
entry:
  %gep1 = getelementptr i8, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i8, ptr %iv2, i64 -1
  %load = load i8, ptr %gep2
  call void @use-byte(i8 %load)
  store i8 %val, ptr %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i8, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-halfword(ptr %array, i64 %count, i16 %val) nounwind {
; CHECK-LABEL: store-post-indexed-halfword
; CHECK: strh w{{[0-9]+}}, [x{{[0-9]+}}], #8
entry:
  %gep1 = getelementptr i16, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i16, ptr %iv2, i64 -1
  %load = load i16, ptr %gep2
  call void @use-halfword(i16 %load)
  store i16 %val, ptr %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i16, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-word(ptr %array, i64 %count, i32 %val) nounwind {
; CHECK-LABEL: store-post-indexed-word
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}], #16
entry:
  %gep1 = getelementptr i32, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i32, ptr %iv2, i64 -1
  %load = load i32, ptr %gep2
  call void @use-word(i32 %load)
  store i32 %val, ptr %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i32, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-doubleword(ptr %array, i64 %count, i64 %val) nounwind {
; CHECK-LABEL: store-post-indexed-doubleword
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}], #32
entry:
  %gep1 = getelementptr i64, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i64, ptr %iv2, i64 -1
  %load = load i64, ptr %gep2
  call void @use-doubleword(i64 %load)
  store i64 %val, ptr %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i64, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-quadword(ptr %array, i64 %count, <2 x i64> %val) nounwind {
; CHECK-LABEL: store-post-indexed-quadword
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}], #64
entry:
  %gep1 = getelementptr <2 x i64>, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr <2 x i64>, ptr %iv2, i64 -1
  %load = load <2 x i64>, ptr %gep2
  call void @use-quadword(<2 x i64> %load)
  store <2 x i64> %val, ptr %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr <2 x i64>, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-float(ptr %array, i64 %count, float %val) nounwind {
; CHECK-LABEL: store-post-indexed-float
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}], #16
entry:
  %gep1 = getelementptr float, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr float, ptr %iv2, i64 -1
  %load = load float, ptr %gep2
  call void @use-float(float %load)
  store float %val, ptr %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr float, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-double(ptr %array, i64 %count, double %val) nounwind {
; CHECK-LABEL: store-post-indexed-double
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}], #32
entry:
  %gep1 = getelementptr double, ptr %array, i64 2
  br label %body

body:
  %iv2 = phi ptr [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr double, ptr %iv2, i64 -1
  %load = load double, ptr %gep2
  call void @use-double(double %load)
  store double %val, ptr %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr double, ptr %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

declare void @use-byte(i8)
declare void @use-halfword(i16)
declare void @use-word(i32)
declare void @use-doubleword(i64)
declare void @use-quadword(<2 x i64>)
declare void @use-float(float)
declare void @use-double(double)

; Check the following transform:
;
; stp w0, [x20]
;  ...
; add x20, x20, #32
;  ->
; stp w0, [x20], #32

define void @store-pair-post-indexed-word() nounwind {
; CHECK-LABEL: store-pair-post-indexed-word
; CHECK: stp w{{[0-9]+}}, w{{[0-9]+}}, [sp], #16
; CHECK: ret
  %src = alloca { i32, i32 }, align 8
  %dst = alloca { i32, i32 }, align 8

  %src.realp = getelementptr inbounds { i32, i32 }, ptr %src, i32 0, i32 0
  %src.real = load i32, ptr %src.realp
  %src.imagp = getelementptr inbounds { i32, i32 }, ptr %src, i32 0, i32 1
  %src.imag = load i32, ptr %src.imagp

  %dst.realp = getelementptr inbounds { i32, i32 }, ptr %dst, i32 0, i32 0
  %dst.imagp = getelementptr inbounds { i32, i32 }, ptr %dst, i32 0, i32 1
  store i32 %src.real, ptr %dst.realp
  store i32 %src.imag, ptr %dst.imagp
  ret void
}

define void @store-pair-post-indexed-doubleword() nounwind {
; CHECK-LABEL: store-pair-post-indexed-doubleword
; CHECK: stp x{{[0-9]+}}, x{{[0-9]+}}, [sp], #32
; CHECK: ret
  %src = alloca { i64, i64 }, align 8
  %dst = alloca { i64, i64 }, align 8

  %src.realp = getelementptr inbounds { i64, i64 }, ptr %src, i32 0, i32 0
  %src.real = load i64, ptr %src.realp
  %src.imagp = getelementptr inbounds { i64, i64 }, ptr %src, i32 0, i32 1
  %src.imag = load i64, ptr %src.imagp

  %dst.realp = getelementptr inbounds { i64, i64 }, ptr %dst, i32 0, i32 0
  %dst.imagp = getelementptr inbounds { i64, i64 }, ptr %dst, i32 0, i32 1
  store i64 %src.real, ptr %dst.realp
  store i64 %src.imag, ptr %dst.imagp
  ret void
}

define void @store-pair-post-indexed-float() nounwind {
; CHECK-LABEL: store-pair-post-indexed-float
; CHECK: stp s{{[0-9]+}}, s{{[0-9]+}}, [sp], #16
; CHECK: ret
  %src = alloca { float, float }, align 8
  %dst = alloca { float, float }, align 8

  %src.realp = getelementptr inbounds { float, float }, ptr %src, i32 0, i32 0
  %src.real = load float, ptr %src.realp
  %src.imagp = getelementptr inbounds { float, float }, ptr %src, i32 0, i32 1
  %src.imag = load float, ptr %src.imagp

  %dst.realp = getelementptr inbounds { float, float }, ptr %dst, i32 0, i32 0
  %dst.imagp = getelementptr inbounds { float, float }, ptr %dst, i32 0, i32 1
  store float %src.real, ptr %dst.realp
  store float %src.imag, ptr %dst.imagp
  ret void
}

define void @store-pair-post-indexed-double() nounwind {
; CHECK-LABEL: store-pair-post-indexed-double
; CHECK: stp d{{[0-9]+}}, d{{[0-9]+}}, [sp], #32
; CHECK: ret
  %src = alloca { double, double }, align 8
  %dst = alloca { double, double }, align 8

  %src.realp = getelementptr inbounds { double, double }, ptr %src, i32 0, i32 0
  %src.real = load double, ptr %src.realp
  %src.imagp = getelementptr inbounds { double, double }, ptr %src, i32 0, i32 1
  %src.imag = load double, ptr %src.imagp

  %dst.realp = getelementptr inbounds { double, double }, ptr %dst, i32 0, i32 0
  %dst.imagp = getelementptr inbounds { double, double }, ptr %dst, i32 0, i32 1
  store double %src.real, ptr %dst.realp
  store double %src.imag, ptr %dst.imagp
  ret void
}

; Check the following transform:
;
; (ldr|str) X, [x20]
;  ...
; sub x20, x20, #16
;  ->
; (ldr|str) X, [x20], #-16
;
; with X being either w0, x0, s0, d0 or q0.

define void @post-indexed-sub-word(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-word
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}], #-8
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}], #-8
  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr i32, ptr %phi1, i64 -1
  %load1 = load i32, ptr %gep1
  %gep2 = getelementptr i32, ptr %phi2, i64 -1
  store i32 %load1, ptr %gep2
  %load2 = load i32, ptr %phi1
  store i32 %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr i32, ptr %phi2, i64 -2
  %gep4 = getelementptr i32, ptr %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-doubleword(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-doubleword
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}], #-16
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}], #-16
  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr i64, ptr %phi1, i64 -1
  %load1 = load i64, ptr %gep1
  %gep2 = getelementptr i64, ptr %phi2, i64 -1
  store i64 %load1, ptr %gep2
  %load2 = load i64, ptr %phi1
  store i64 %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr i64, ptr %phi2, i64 -2
  %gep4 = getelementptr i64, ptr %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-quadword(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-quadword
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}], #-32
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}], #-32
  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr <2 x i64>, ptr %phi1, i64 -1
  %load1 = load <2 x i64>, ptr %gep1
  %gep2 = getelementptr <2 x i64>, ptr %phi2, i64 -1
  store <2 x i64> %load1, ptr %gep2
  %load2 = load <2 x i64>, ptr %phi1
  store <2 x i64> %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr <2 x i64>, ptr %phi2, i64 -2
  %gep4 = getelementptr <2 x i64>, ptr %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-float(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-float
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}], #-8
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}], #-8
  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr float, ptr %phi1, i64 -1
  %load1 = load float, ptr %gep1
  %gep2 = getelementptr float, ptr %phi2, i64 -1
  store float %load1, ptr %gep2
  %load2 = load float, ptr %phi1
  store float %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr float, ptr %phi2, i64 -2
  %gep4 = getelementptr float, ptr %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-double(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-double
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}], #-16
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}], #-16
  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr double, ptr %phi1, i64 -1
  %load1 = load double, ptr %gep1
  %gep2 = getelementptr double, ptr %phi2, i64 -1
  store double %load1, ptr %gep2
  %load2 = load double, ptr %phi1
  store double %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr double, ptr %phi2, i64 -2
  %gep4 = getelementptr double, ptr %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-doubleword-offset-min(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-doubleword-offset-min
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}], #-256
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}], #-256
  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr i64, ptr %phi1, i64 1
  %load1 = load i64, ptr %gep1
  %gep2 = getelementptr i64, ptr %phi2, i64 1
  store i64 %load1, ptr %gep2
  %load2 = load i64, ptr %phi1
  store i64 %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr i64, ptr %phi2, i64 -32
  %gep4 = getelementptr i64, ptr %phi1, i64 -32
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-doubleword-offset-out-of-range(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-doubleword-offset-out-of-range
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}]
; CHECK: add x{{[0-9]+}}, x{{[0-9]+}}, #256
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}]
; CHECK: add x{{[0-9]+}}, x{{[0-9]+}}, #256

  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr i64, ptr %phi1, i64 1
  %load1 = load i64, ptr %gep1
  %gep2 = getelementptr i64, ptr %phi2, i64 1
  store i64 %load1, ptr %gep2
  %load2 = load i64, ptr %phi1
  store i64 %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr i64, ptr %phi2, i64 32
  %gep4 = getelementptr i64, ptr %phi1, i64 32
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-paired-min-offset(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-paired-min-offset
; CHECK: ldp x{{[0-9]+}}, x{{[0-9]+}}, [x{{[0-9]+}}], #-512
; CHECK: stp x{{[0-9]+}}, x{{[0-9]+}}, [x{{[0-9]+}}], #-512
  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr i64, ptr %phi1, i64 1
  %load1 = load i64, ptr %gep1
  %gep2 = getelementptr i64, ptr %phi2, i64 1
  %load2 = load i64, ptr %phi1
  store i64 %load1, ptr %gep2
  store i64 %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr i64, ptr %phi2, i64 -64
  %gep4 = getelementptr i64, ptr %phi1, i64 -64
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-paired-offset-out-of-range(ptr %a, ptr %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-paired-offset-out-of-range
; CHECK: ldp x{{[0-9]+}}, x{{[0-9]+}}, [x{{[0-9]+}}]
; CHECK: add x{{[0-9]+}}, x{{[0-9]+}}, #512
; CHECK: stp x{{[0-9]+}}, x{{[0-9]+}}, [x{{[0-9]+}}]
; CHECK: add x{{[0-9]+}}, x{{[0-9]+}}, #512
  br label %for.body
for.body:
  %phi1 = phi ptr [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi ptr [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr i64, ptr %phi1, i64 1
  %load1 = load i64, ptr %phi1
  %gep2 = getelementptr i64, ptr %phi2, i64 1
  %load2 = load i64, ptr %gep1
  store i64 %load1, ptr %gep2
  store i64 %load2, ptr %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr i64, ptr %phi2, i64 64
  %gep4 = getelementptr i64, ptr %phi1, i64 64
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

; DAGCombiner::MergeConsecutiveStores merges this into a vector store,
; replaceZeroVectorStore should split the vector store back into
; scalar stores which should get merged by AArch64LoadStoreOptimizer.
define void @merge_zr32(ptr %p) {
; CHECK-LABEL: merge_zr32:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: str xzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store i32 0, ptr %p
  %p1 = getelementptr i32, ptr %p, i32 1
  store i32 0, ptr %p1
  ret void
}

; Same as merge_zr32 but the merged stores should also get paried.
define void @merge_zr32_2(ptr %p) {
; CHECK-LABEL: merge_zr32_2:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: stp xzr, xzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}, #8]
; CHECK-NEXT: ret
entry:
  store i32 0, ptr %p
  %p1 = getelementptr i32, ptr %p, i32 1
  store i32 0, ptr %p1
  %p2 = getelementptr i32, ptr %p, i64 2
  store i32 0, ptr %p2
  %p3 = getelementptr i32, ptr %p, i64 3
  store i32 0, ptr %p3
  ret void
}

; Like merge_zr32_2, but checking the largest allowed stp immediate offset.
define void @merge_zr32_2_offset(ptr %p) {
; CHECK-LABEL: merge_zr32_2_offset:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: stp xzr, xzr, [x{{[0-9]+}}, #504]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #504]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #508]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #512]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #516]
; CHECK-NEXT: ret
entry:
  %p0 = getelementptr i32, ptr %p, i32 126
  store i32 0, ptr %p0
  %p1 = getelementptr i32, ptr %p, i32 127
  store i32 0, ptr %p1
  %p2 = getelementptr i32, ptr %p, i64 128
  store i32 0, ptr %p2
  %p3 = getelementptr i32, ptr %p, i64 129
  store i32 0, ptr %p3
  ret void
}

; Like merge_zr32, but replaceZeroVectorStore should not split this
; vector store since the address offset is too large for the stp
; instruction.
define void @no_merge_zr32_2_offset(ptr %p) {
; CHECK-LABEL: no_merge_zr32_2_offset:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: movi v[[REG:[0-9]]].2d, #0000000000000000
; NOSTRICTALIGN-NEXT: str q[[REG]], [x{{[0-9]+}}, #4096]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #4096]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #4100]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #4104]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #4108]
; CHECK-NEXT: ret
entry:
  %p0 = getelementptr i32, ptr %p, i32 1024
  store i32 0, ptr %p0
  %p1 = getelementptr i32, ptr %p, i32 1025
  store i32 0, ptr %p1
  %p2 = getelementptr i32, ptr %p, i64 1026
  store i32 0, ptr %p2
  %p3 = getelementptr i32, ptr %p, i64 1027
  store i32 0, ptr %p3
  ret void
}

; Like merge_zr32, but replaceZeroVectorStore should not split the
; vector store since the zero constant vector has multiple uses, so we
; err on the side that allows for stp q instruction generation.
define void @merge_zr32_3(ptr %p) {
; CHECK-LABEL: merge_zr32_3:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: movi v[[REG:[0-9]]].2d, #0000000000000000
; NOSTRICTALIGN-NEXT: stp q[[REG]], q[[REG]], [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}, #8]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}, #16]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}, #24]
; CHECK-NEXT: ret
entry:
  store i32 0, ptr %p
  %p1 = getelementptr i32, ptr %p, i32 1
  store i32 0, ptr %p1
  %p2 = getelementptr i32, ptr %p, i64 2
  store i32 0, ptr %p2
  %p3 = getelementptr i32, ptr %p, i64 3
  store i32 0, ptr %p3
  %p4 = getelementptr i32, ptr %p, i64 4
  store i32 0, ptr %p4
  %p5 = getelementptr i32, ptr %p, i64 5
  store i32 0, ptr %p5
  %p6 = getelementptr i32, ptr %p, i64 6
  store i32 0, ptr %p6
  %p7 = getelementptr i32, ptr %p, i64 7
  store i32 0, ptr %p7
  ret void
}

; Like merge_zr32, but with 2-vector type.
define void @merge_zr32_2vec(ptr %p) {
; CHECK-LABEL: merge_zr32_2vec:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: str xzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store <2 x i32> zeroinitializer, ptr %p
  ret void
}

; Like merge_zr32, but with 3-vector type.
define void @merge_zr32_3vec(ptr %p) {
; CHECK-LABEL: merge_zr32_3vec:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}, #8]
; NOSTRICTALIGN-NEXT: str xzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}, #4]
; STRICTALIGN-NEXT: str wzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store <3 x i32> zeroinitializer, ptr %p
  ret void
}

; Like merge_zr32, but with 4-vector type.
define void @merge_zr32_4vec(ptr %p) {
; CHECK-LABEL: merge_zr32_4vec:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: stp xzr, xzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}, #8]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store <4 x i32> zeroinitializer, ptr %p
  ret void
}

; Like merge_zr32, but with 2-vector float type.
define void @merge_zr32_2vecf(ptr %p) {
; CHECK-LABEL: merge_zr32_2vecf:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: str xzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store <2 x float> zeroinitializer, ptr %p
  ret void
}

; Like merge_zr32, but with 4-vector float type.
define void @merge_zr32_4vecf(ptr %p) {
; CHECK-LABEL: merge_zr32_4vecf:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: stp xzr, xzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}, #8]
; STRICTALIGN-NEXT: stp wzr, wzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store <4 x float> zeroinitializer, ptr %p
  ret void
}

; Similar to merge_zr32, but for 64-bit values.
define void @merge_zr64(ptr %p) {
; CHECK-LABEL: merge_zr64:
; CHECK: // %entry
; CHECK-NEXT: stp xzr, xzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store i64 0, ptr %p
  %p1 = getelementptr i64, ptr %p, i64 1
  store i64 0, ptr %p1
  ret void
}

; Similar to merge_zr32, but for 64-bit values and with unaligned stores.
define void @merge_zr64_unalign(ptr %p) {
; CHECK-LABEL: merge_zr64_unalign:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: stp xzr, xzr, [x{{[0-9]+}}]
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; STRICTALIGN: strb
; CHECK-NEXT: ret
entry:
  store <2 x i64> zeroinitializer, ptr %p, align 1
  ret void
}

; Similar to merge_zr32_3, replaceZeroVectorStore should not split the
; vector store since the zero constant vector has multiple uses.
define void @merge_zr64_2(ptr %p) {
; CHECK-LABEL: merge_zr64_2:
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: movi v[[REG:[0-9]]].2d, #0000000000000000
; NOSTRICTALIGN-NEXT: stp q[[REG]], q[[REG]], [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp xzr, xzr, [x{{[0-9]+}}]
; STRICTALIGN-NEXT: stp xzr, xzr, [x{{[0-9]+}}, #16]
; CHECK-NEXT: ret
entry:
  store i64 0, ptr %p
  %p1 = getelementptr i64, ptr %p, i64 1
  store i64 0, ptr %p1
  %p2 = getelementptr i64, ptr %p, i64 2
  store i64 0, ptr %p2
  %p3 = getelementptr i64, ptr %p, i64 3
  store i64 0, ptr %p3
  ret void
}

; Like merge_zr64, but with 2-vector double type.
define void @merge_zr64_2vecd(ptr %p) {
; CHECK-LABEL: merge_zr64_2vecd:
; CHECK: // %entry
; CHECK-NEXT: stp xzr, xzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store <2 x double> zeroinitializer, ptr %p
  ret void
}

; Like merge_zr64, but with 3-vector i64 type.
define void @merge_zr64_3vec(ptr %p) {
; CHECK-LABEL: merge_zr64_3vec:
; CHECK: // %entry
; CHECK-NEXT: stp xzr, xzr, [x{{[0-9]+}}, #8]
; CHECK-NEXT: str xzr, [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store <3 x i64> zeroinitializer, ptr %p
  ret void
}

; Like merge_zr64_2, but with 4-vector double type.
define void @merge_zr64_4vecd(ptr %p) {
; CHECK-LABEL: merge_zr64_4vecd:
; CHECK: // %entry
; CHECK-NEXT: movi v[[REG:[0-9]]].2d, #0000000000000000
; CHECK-NEXT: stp q[[REG]], q[[REG]], [x{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  store <4 x double> zeroinitializer, ptr %p
  ret void
}

; Verify that non-consecutive merges do not generate q0
define void @merge_multiple_128bit_stores(ptr %p) {
; CHECK-LABEL: merge_multiple_128bit_stores
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: movi v[[REG:[0-9]]].2d, #0000000000000000
; NOSTRICTALIGN-NEXT: str q0, [x0]
; NOSTRICTALIGN-NEXT: stur q0, [x0, #24]
; NOSTRICTALIGN-NEXT: str q0, [x0, #48]
; STRICTALIGN-NEXT: stp xzr, xzr, [x0]
; STRICTALIGN-NEXT: stp xzr, xzr, [x0, #24]
; STRICTALIGN-NEXT: stp xzr, xzr, [x0, #48]
; CHECK-NEXT: ret
entry:
  store i64 0, ptr %p
  %p1 = getelementptr i64, ptr %p, i64 1
  store i64 0, ptr %p1
  %p3 = getelementptr i64, ptr %p, i64 3
  store i64 0, ptr %p3
  %p4 = getelementptr i64, ptr %p, i64 4
  store i64 0, ptr %p4
  %p6 = getelementptr i64, ptr %p, i64 6
  store i64 0, ptr %p6
  %p7 = getelementptr i64, ptr %p, i64 7
  store i64 0, ptr %p7
  ret void
}

; Verify that large stores generate stp q
define void @merge_multiple_128bit_stores_consec(ptr %p) {
; CHECK-LABEL: merge_multiple_128bit_stores_consec
; CHECK: // %entry
; NOSTRICTALIGN-NEXT: movi v[[REG:[0-9]]].2d, #0000000000000000
; NOSTRICTALIGN-NEXT: stp q[[REG]], q[[REG]], [x{{[0-9]+}}]
; NOSTRICTALIGN-NEXT: stp q[[REG]], q[[REG]], [x{{[0-9]+}}, #32]
; STRICTALIGN-NEXT: stp	 xzr, xzr, [x0]
; STRICTALIGN-NEXT: stp	 xzr, xzr, [x0, #16]
; STRICTALIGN-NEXT: stp	 xzr, xzr, [x0, #32]
; STRICTALIGN-NEXT: stp  xzr, xzr, [x0, #48]
; CHECK-NEXT: ret
entry:
  store i64 0, ptr %p
  %p1 = getelementptr i64, ptr %p, i64 1
  store i64 0, ptr %p1
  %p2 = getelementptr i64, ptr %p, i64 2
  store i64 0, ptr %p2
  %p3 = getelementptr i64, ptr %p, i64 3
  store i64 0, ptr %p3
  %p4 = getelementptr i64, ptr %p, i64 4
  store i64 0, ptr %p4
  %p5 = getelementptr i64, ptr %p, i64 5
  store i64 0, ptr %p5
  %p6 = getelementptr i64, ptr %p, i64 6
  store i64 0, ptr %p6
  %p7 = getelementptr i64, ptr %p, i64 7
  store i64 0, ptr %p7
  ret void
}

; Check for bug 34674 where invalid add of xzr was being generated.
; CHECK-LABEL: bug34674:
; CHECK: // %entry
; CHECK-NEXT: mov [[ZREG:x[0-9]+]], xzr
; CHECK-NEXT: mov x8, x0
; CHECK-NEXT: add x0, [[ZREG]], #1
; CHECK-NEXT: stp xzr, xzr, [x8]
define i64 @bug34674(ptr %p) {
entry:
  store <2 x i64> zeroinitializer, ptr %p
  %ld = load i64, ptr %p
  %add = add i64 %ld, 1
  ret i64 %add
}

; CHECK-LABEL: trunc_splat_zero:
; CHECK-DAG: strh wzr, [x0]
define void @trunc_splat_zero(ptr %ptr) {
  store <2 x i8> zeroinitializer, ptr %ptr, align 2
  ret void
}

; CHECK-LABEL: trunc_splat:
; CHECK: mov [[VAL:w[0-9]+]], #42
; CHECK: movk [[VAL]], #42, lsl #16
; CHECK: str [[VAL]], [x0]
define void @trunc_splat(ptr %ptr) {
  store <2 x i16> <i16 42, i16 42>, ptr %ptr, align 4
  ret void
}
