; RUN: llc -march=hexagon < %s
; Used to fail with "Cannot select: 0x16cf370: v2i16,ch = load"

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

%struct.ext_hdrs.10.65.142.274.307.318.329.681.692.703.714.725.736.758.791.802.846.857.868.879.890.901.945.956.958 = type { i8, i8, i8, i8, i8, i8, i16, i32, [8 x %struct.hcdc_ext_vec.9.64.141.273.306.317.328.680.691.702.713.724.735.757.790.801.845.856.867.878.889.900.944.955.957] }
%struct.hcdc_ext_vec.9.64.141.273.306.317.328.680.691.702.713.724.735.757.790.801.845.856.867.878.889.900.944.955.957 = type { i8, i8, i16 }

define void @foo(ptr %hc_ext_info) nounwind {
entry:
  br i1 undef, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  br i1 undef, label %if.end5, label %if.then3

if.then3:                                         ; preds = %if.end
  br label %if.end5

if.end5:                                          ; preds = %if.then3, %if.end
  %add.ptr = getelementptr inbounds %struct.ext_hdrs.10.65.142.274.307.318.329.681.692.703.714.725.736.758.791.802.846.857.868.879.890.901.945.956.958, ptr %hc_ext_info, i32 0, i32 8, i32 0
  %add.ptr22 = getelementptr inbounds %struct.ext_hdrs.10.65.142.274.307.318.329.681.692.703.714.725.736.758.791.802.846.857.868.879.890.901.945.956.958, ptr null, i32 0, i32 8, i32 undef
  br label %while.cond

while.cond:                                       ; preds = %if.end419, %if.end5
  %gre_chksum.0 = phi <2 x i8> [ undef, %if.end5 ], [ %gre_chksum.2, %if.end419 ]
  %cmp23 = icmp ult ptr null, %add.ptr
  %cmp25 = icmp ult ptr null, %add.ptr22
  %sel1 = and i1 %cmp23, %cmp25
  br i1 %sel1, label %while.body, label %while.end422

while.body:                                       ; preds = %while.cond
  switch i8 undef, label %if.end419 [
    i8 5, label %if.then70
    i8 3, label %if.then70
    i8 2, label %if.then70
    i8 1, label %if.then70
    i8 0, label %if.then70
    i8 4, label %if.then93
    i8 6, label %if.then195
  ]

if.then70:                                        ; preds = %while.body, %while.body, %while.body, %while.body, %while.body
  unreachable

if.then93:                                        ; preds = %while.body
  unreachable

if.then195:                                       ; preds = %while.body
  br i1 undef, label %if.end274, label %if.then202

if.then202:                                       ; preds = %if.then195
  br label %while.body222

while.body222:                                    ; preds = %while.body222, %if.then202
  br i1 undef, label %if.end240, label %while.body222

if.end240:                                        ; preds = %while.body222
  %_p_vec_full100 = load <2 x i8>, ptr undef, align 8
  br label %if.end274

if.end274:                                        ; preds = %if.end240, %if.then195
  %gre_chksum.1 = phi <2 x i8> [ %gre_chksum.0, %if.then195 ], [ %_p_vec_full100, %if.end240 ]
  br label %if.end419

if.end419:                                        ; preds = %if.end274, %while.body
  %gre_chksum.2 = phi <2 x i8> [ %gre_chksum.0, %while.body ], [ %gre_chksum.1, %if.end274 ]
  br label %while.cond

while.end422:                                     ; preds = %while.cond
  ret void
}
