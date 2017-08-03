; RUN: opt -S -force-vector-width=2 -force-vector-interleave=1 -loop-vectorize -verify-loop-info -simplifycfg < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test predication of non-void instructions, specifically (i) that these
; instructions permit vectorization and (ii) the creation of an insertelement
; and a Phi node. We check the full 2-element sequence for the first
; instruction; For the rest we'll just make sure they get predicated based
; on the code generated for the first element.
define void @test(i32* nocapture %asd, i32* nocapture %aud,
                  i32* nocapture %asr, i32* nocapture %aur) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %if.end
  ret void

; CHECK-LABEL: test
; CHECK: vector.body:
; CHECK:   %[[SDEE:[a-zA-Z0-9]+]] = extractelement <2 x i1> %{{.*}}, i32 0
; CHECK:   %[[SDCC:[a-zA-Z0-9]+]] = icmp eq i1 %[[SDEE]], true
; CHECK:   br i1 %[[SDCC]], label %[[CSD:[a-zA-Z0-9.]+]], label %[[ESD:[a-zA-Z0-9.]+]]
; CHECK: [[CSD]]:
; CHECK:   %[[SDA0:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 0
; CHECK:   %[[SDA1:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 0
; CHECK:   %[[SD0:[a-zA-Z0-9]+]] = sdiv i32 %[[SDA0]], %[[SDA1]]
; CHECK:   %[[SD1:[a-zA-Z0-9]+]] = insertelement <2 x i32> undef, i32 %[[SD0]], i32 0
; CHECK:   br label %[[ESD]]
; CHECK: [[ESD]]:
; CHECK:   %[[SDR:[a-zA-Z0-9]+]] = phi <2 x i32> [ undef, %vector.body ], [ %[[SD1]], %[[CSD]] ]
; CHECK:   %[[SDEEH:[a-zA-Z0-9]+]] = extractelement <2 x i1> %{{.*}}, i32 1
; CHECK:   %[[SDCCH:[a-zA-Z0-9]+]] = icmp eq i1 %[[SDEEH]], true
; CHECK:   br i1 %[[SDCCH]], label %[[CSDH:[a-zA-Z0-9.]+]], label %[[ESDH:[a-zA-Z0-9.]+]]
; CHECK: [[CSDH]]:
; CHECK:   %[[SDA0H:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 1
; CHECK:   %[[SDA1H:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 1
; CHECK:   %[[SD0H:[a-zA-Z0-9]+]] = sdiv i32 %[[SDA0H]], %[[SDA1H]]
; CHECK:   %[[SD1H:[a-zA-Z0-9]+]] = insertelement <2 x i32> %[[SDR]], i32 %[[SD0H]], i32 1
; CHECK:   br label %[[ESDH]]
; CHECK: [[ESDH]]:
; CHECK:   %{{.*}} = phi <2 x i32> [ %[[SDR]], %[[ESD]] ], [ %[[SD1H]], %[[CSDH]] ]

; CHECK:   %[[UDEE:[a-zA-Z0-9]+]] = extractelement <2 x i1> %{{.*}}, i32 0
; CHECK:   %[[UDCC:[a-zA-Z0-9]+]] = icmp eq i1 %[[UDEE]], true
; CHECK:   br i1 %[[UDCC]], label %[[CUD:[a-zA-Z0-9.]+]], label %[[EUD:[a-zA-Z0-9.]+]]
; CHECK: [[CUD]]:
; CHECK:   %[[UDA0:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 0
; CHECK:   %[[UDA1:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 0
; CHECK:   %[[UD0:[a-zA-Z0-9]+]] = udiv i32 %[[UDA0]], %[[UDA1]]
; CHECK:   %[[UD1:[a-zA-Z0-9]+]] = insertelement <2 x i32> undef, i32 %[[UD0]], i32 0
; CHECK:   br label %[[EUD]]
; CHECK: [[EUD]]:
; CHECK:   %{{.*}} = phi <2 x i32> [ undef, %{{.*}} ], [ %[[UD1]], %[[CUD]] ]

; CHECK:   %[[SREE:[a-zA-Z0-9]+]] = extractelement <2 x i1> %{{.*}}, i32 0
; CHECK:   %[[SRCC:[a-zA-Z0-9]+]] = icmp eq i1 %[[SREE]], true
; CHECK:   br i1 %[[SRCC]], label %[[CSR:[a-zA-Z0-9.]+]], label %[[ESR:[a-zA-Z0-9.]+]]
; CHECK: [[CSR]]:
; CHECK:   %[[SRA0:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 0
; CHECK:   %[[SRA1:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 0
; CHECK:   %[[SR0:[a-zA-Z0-9]+]] = srem i32 %[[SRA0]], %[[SRA1]]
; CHECK:   %[[SR1:[a-zA-Z0-9]+]] = insertelement <2 x i32> undef, i32 %[[SR0]], i32 0
; CHECK:   br label %[[ESR]]
; CHECK: [[ESR]]:
; CHECK:   %{{.*}} = phi <2 x i32> [ undef, %{{.*}} ], [ %[[SR1]], %[[CSR]] ]

; CHECK:   %[[UREE:[a-zA-Z0-9]+]] = extractelement <2 x i1> %{{.*}}, i32 0
; CHECK:   %[[URCC:[a-zA-Z0-9]+]] = icmp eq i1 %[[UREE]], true
; CHECK:   br i1 %[[URCC]], label %[[CUR:[a-zA-Z0-9.]+]], label %[[EUR:[a-zA-Z0-9.]+]]
; CHECK: [[CUR]]:
; CHECK:   %[[URA0:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 0
; CHECK:   %[[URA1:[a-zA-Z0-9]+]] = extractelement <2 x i32> %{{.*}}, i32 0
; CHECK:   %[[UR0:[a-zA-Z0-9]+]] = urem i32 %[[URA0]], %[[URA1]]
; CHECK:   %[[UR1:[a-zA-Z0-9]+]] = insertelement <2 x i32> undef, i32 %[[UR0]], i32 0
; CHECK:   br label %[[EUR]]
; CHECK: [[EUR]]:
; CHECK:   %{{.*}} = phi <2 x i32> [ undef, %{{.*}} ], [ %[[UR1]], %[[CUR]] ]

for.body:                                         ; preds = %if.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end ]
  %isd = getelementptr inbounds i32, i32* %asd, i64 %indvars.iv
  %iud = getelementptr inbounds i32, i32* %aud, i64 %indvars.iv
  %isr = getelementptr inbounds i32, i32* %asr, i64 %indvars.iv
  %iur = getelementptr inbounds i32, i32* %aur, i64 %indvars.iv
  %lsd = load i32, i32* %isd, align 4
  %lud = load i32, i32* %iud, align 4
  %lsr = load i32, i32* %isr, align 4
  %lur = load i32, i32* %iur, align 4
  %psd = add nsw i32 %lsd, 23
  %pud = add nsw i32 %lud, 24
  %psr = add nsw i32 %lsr, 25
  %pur = add nsw i32 %lur, 26
  %cmp1 = icmp slt i32 %lsd, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %rsd = sdiv i32 %psd, %lsd
  %rud = udiv i32 %pud, %lud
  %rsr = srem i32 %psr, %lsr
  %rur = urem i32 %pur, %lur
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %ysd.0 = phi i32 [ %rsd, %if.then ], [ %psd, %for.body ]
  %yud.0 = phi i32 [ %rud, %if.then ], [ %pud, %for.body ]
  %ysr.0 = phi i32 [ %rsr, %if.then ], [ %psr, %for.body ]
  %yur.0 = phi i32 [ %rur, %if.then ], [ %pur, %for.body ]
  store i32 %ysd.0, i32* %isd, align 4
  store i32 %yud.0, i32* %iud, align 4
  store i32 %ysr.0, i32* %isr, align 4
  store i32 %yur.0, i32* %iur, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 128
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @test_scalar2scalar(i32* nocapture %asd, i32* nocapture %bsd) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %if.end
  ret void

; CHECK-LABEL: test_scalar2scalar
; CHECK: vector.body:
; CHECK:   br i1 %{{.*}}, label %[[THEN:[a-zA-Z0-9.]+]], label %[[FI:[a-zA-Z0-9.]+]]
; CHECK: [[THEN]]:
; CHECK:   %[[PD:[a-zA-Z0-9]+]] = sdiv i32 %{{.*}}, %{{.*}}
; CHECK:   br label %[[FI]]
; CHECK: [[FI]]:
; CHECK:   %{{.*}} = phi i32 [ undef, %vector.body ], [ %[[PD]], %[[THEN]] ]

for.body:                                         ; preds = %if.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end ]
  %isd = getelementptr inbounds i32, i32* %asd, i64 %indvars.iv
  %lsd = load i32, i32* %isd, align 4
  %isd.b = getelementptr inbounds i32, i32* %bsd, i64 %indvars.iv
  %lsd.b = load i32, i32* %isd.b, align 4
  %psd = add nsw i32 %lsd, 23
  %cmp1 = icmp slt i32 %lsd, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %sd1 = sdiv i32 %psd, %lsd
  %rsd = sdiv i32 %lsd.b, %sd1
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %ysd.0 = phi i32 [ %rsd, %if.then ], [ %psd, %for.body ]
  store i32 %ysd.0, i32* %isd, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 128
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @pr30172(i32* nocapture %asd, i32* nocapture %bsd) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %if.end
  ret void

; CHECK-LABEL: pr30172
; CHECK: vector.body:
; CHECK: %[[CMP1:.+]] = icmp slt <2 x i32> %[[VAL:.+]], <i32 100, i32 100>
; CHECK: %[[CMP2:.+]] = icmp sge <2 x i32> %[[VAL]], <i32 200, i32 200>
; CHECK: %[[XOR:.+]] = xor <2 x i1> %[[CMP1]], <i1 true, i1 true>
; CHECK: %[[AND1:.+]] = and <2 x i1> %[[XOR]], <i1 true, i1 true>
; CHECK: %[[OR1:.+]] = or <2 x i1> zeroinitializer, %[[AND1]]
; CHECK: %[[AND2:.+]] = and <2 x i1> %[[CMP2]], %[[OR1]]
; CHECK: %[[OR2:.+]] = or <2 x i1> zeroinitializer, %[[AND2]]
; CHECK: %[[AND3:.+]] = and <2 x i1> %[[CMP1]], <i1 true, i1 true>
; CHECK: %[[OR3:.+]] = or <2 x i1> %[[OR2]], %[[AND3]]
; CHECK: %[[EXTRACT:.+]] = extractelement <2 x i1> %[[OR3]], i32 0
; CHECK: %[[MASK:.+]] = icmp eq i1 %[[EXTRACT]], true
; CHECK: br i1 %[[MASK]], label %[[THEN:[a-zA-Z0-9.]+]], label %[[FI:[a-zA-Z0-9.]+]]
; CHECK: [[THEN]]:
; CHECK:   %[[PD:[a-zA-Z0-9]+]] = sdiv i32 %{{.*}}, %{{.*}}
; CHECK:   br label %[[FI]]
; CHECK: [[FI]]:
; CHECK:   %{{.*}} = phi i32 [ undef, %vector.body ], [ %[[PD]], %[[THEN]] ]


for.body:                                         ; preds = %if.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end ]
  %isd = getelementptr inbounds i32, i32* %asd, i64 %indvars.iv
  %lsd = load i32, i32* %isd, align 4
  %isd.b = getelementptr inbounds i32, i32* %bsd, i64 %indvars.iv
  %lsd.b = load i32, i32* %isd.b, align 4
  %psd = add nsw i32 %lsd, 23
  %cmp1 = icmp slt i32 %lsd, 100
  br i1 %cmp1, label %if.then, label %check

check:                                            ; preds = %for.body
  %cmp2 = icmp sge i32 %lsd, 200
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %check, %for.body
  %sd1 = sdiv i32 %psd, %lsd
  %rsd = sdiv i32 %lsd.b, %sd1
  br label %if.end

if.end:                                           ; preds = %if.then, %check
  %ysd.0 = phi i32 [ %rsd, %if.then ], [ %psd, %check ] 
  store i32 %ysd.0, i32* %isd, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 128
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}


define i32 @predicated_udiv_scalarized_operand(i32* %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %for.body

; CHECK-LABEL: predicated_udiv_scalarized_operand
; CHECK: vector.body:
; CHECK:   %wide.load = load <2 x i32>, <2 x i32>* {{.*}}, align 4
; CHECK:   br i1 {{.*}}, label %[[IF0:.+]], label %[[CONT0:.+]]
; CHECK: [[IF0]]:
; CHECK:   %[[T00:.+]] = extractelement <2 x i32> %wide.load, i32 0
; CHECK:   %[[T01:.+]] = extractelement <2 x i32> %wide.load, i32 0
; CHECK:   %[[T02:.+]] = add nsw i32 %[[T01]], %x
; CHECK:   %[[T03:.+]] = udiv i32 %[[T00]], %[[T02]]
; CHECK:   %[[T04:.+]] = insertelement <2 x i32> undef, i32 %[[T03]], i32 0
; CHECK:   br label %[[CONT0]]
; CHECK: [[CONT0]]:
; CHECK:   %[[T05:.+]] = phi <2 x i32> [ undef, %vector.body ], [ %[[T04]], %[[IF0]] ]
; CHECK:   br i1 {{.*}}, label %[[IF1:.+]], label %[[CONT1:.+]]
; CHECK: [[IF1]]:
; CHECK:   %[[T06:.+]] = extractelement <2 x i32> %wide.load, i32 1
; CHECK:   %[[T07:.+]] = extractelement <2 x i32> %wide.load, i32 1
; CHECK:   %[[T08:.+]] = add nsw i32 %[[T07]], %x
; CHECK:   %[[T09:.+]] = udiv i32 %[[T06]], %[[T08]]
; CHECK:   %[[T10:.+]] = insertelement <2 x i32> %[[T05]], i32 %[[T09]], i32 1
; CHECK:   br label %[[CONT1]]
; CHECK: [[CONT1]]:
; CHECK:   phi <2 x i32> [ %[[T05]], %[[CONT0]] ], [ %[[T10]], %[[IF1]] ]
; CHECK:   br i1 {{.*}}, label %middle.block, label %vector.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %r = phi i32 [ 0, %entry ], [ %tmp6, %for.inc ]
  %tmp0 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp2 = load i32, i32* %tmp0, align 4
  br i1 %c, label %if.then, label %for.inc

if.then:
  %tmp3 = add nsw i32 %tmp2, %x
  %tmp4 = udiv i32 %tmp2, %tmp3
  br label %for.inc

for.inc:
  %tmp5 = phi i32 [ %tmp2, %for.body ], [ %tmp4, %if.then]
  %tmp6 = add i32 %r, %tmp5
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp7 = phi i32 [ %tmp6, %for.inc ]
  ret i32 %tmp7
}
