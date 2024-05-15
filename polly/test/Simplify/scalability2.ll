; RUN: opt %loadPolly -polly-ignore-inbounds -polly-print-simplify -disable-output < %s | FileCheck %s -match-full-lines
;
; Test scalability.
;
define void @func(i32 %n, ptr noalias nonnull %A,
i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %p7, i32 %p8, i32 %p9,
i32 %p10, i32 %p11, i32 %p12, i32 %p13, i32 %p14, i32 %p15, i32 %p16, i32 %p17, i32 %p18, i32 %p19,
i32 %p20, i32 %p21, i32 %p22, i32 %p23, i32 %p24, i32 %p25, i32 %p26, i32 %p27, i32 %p28, i32 %p29,
i32 %p30, i32 %p31, i32 %p32, i32 %p33, i32 %p34, i32 %p35, i32 %p36, i32 %p37, i32 %p38, i32 %p39,
i32 %p40, i32 %p41, i32 %p42, i32 %p43, i32 %p44, i32 %p45, i32 %p46, i32 %p47, i32 %p48, i32 %p49,
i32 %p50, i32 %p51, i32 %p52, i32 %p53, i32 %p54, i32 %p55, i32 %p56, i32 %p57, i32 %p58, i32 %p59) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit


    body:
      %A0 = getelementptr inbounds double, ptr %A, i32 %p0
      %A1 = getelementptr inbounds double, ptr %A, i32 %p1
      %A2 = getelementptr inbounds double, ptr %A, i32 %p2
      %A3 = getelementptr inbounds double, ptr %A, i32 %p3
      %A4 = getelementptr inbounds double, ptr %A, i32 %p4
      %A5 = getelementptr inbounds double, ptr %A, i32 %p5
      %A6 = getelementptr inbounds double, ptr %A, i32 %p6
      %A7 = getelementptr inbounds double, ptr %A, i32 %p7
      %A8 = getelementptr inbounds double, ptr %A, i32 %p8
      %A9 = getelementptr inbounds double, ptr %A, i32 %p9
      %A10 = getelementptr inbounds double, ptr %A, i32 %p10
      %A11 = getelementptr inbounds double, ptr %A, i32 %p11
      %A12 = getelementptr inbounds double, ptr %A, i32 %p12
      %A13 = getelementptr inbounds double, ptr %A, i32 %p13
      %A14 = getelementptr inbounds double, ptr %A, i32 %p14
      %A15 = getelementptr inbounds double, ptr %A, i32 %p15
      %A16 = getelementptr inbounds double, ptr %A, i32 %p16
      %A17 = getelementptr inbounds double, ptr %A, i32 %p17
      %A18 = getelementptr inbounds double, ptr %A, i32 %p18
      %A19 = getelementptr inbounds double, ptr %A, i32 %p19
      %A20 = getelementptr inbounds double, ptr %A, i32 %p20
      %A21 = getelementptr inbounds double, ptr %A, i32 %p21
      %A22 = getelementptr inbounds double, ptr %A, i32 %p22
      %A23 = getelementptr inbounds double, ptr %A, i32 %p23
      %A24 = getelementptr inbounds double, ptr %A, i32 %p24
      %A25 = getelementptr inbounds double, ptr %A, i32 %p25
      %A26 = getelementptr inbounds double, ptr %A, i32 %p26
      %A27 = getelementptr inbounds double, ptr %A, i32 %p27
      %A28 = getelementptr inbounds double, ptr %A, i32 %p28
      %A29 = getelementptr inbounds double, ptr %A, i32 %p29
      %A30 = getelementptr inbounds double, ptr %A, i32 %p30
      %A31 = getelementptr inbounds double, ptr %A, i32 %p31
      %A32 = getelementptr inbounds double, ptr %A, i32 %p32
      %A33 = getelementptr inbounds double, ptr %A, i32 %p33
      %A34 = getelementptr inbounds double, ptr %A, i32 %p34
      %A35 = getelementptr inbounds double, ptr %A, i32 %p35
      %A36 = getelementptr inbounds double, ptr %A, i32 %p36
      %A37 = getelementptr inbounds double, ptr %A, i32 %p37
      %A38 = getelementptr inbounds double, ptr %A, i32 %p38
      %A39 = getelementptr inbounds double, ptr %A, i32 %p39
      %A40 = getelementptr inbounds double, ptr %A, i32 %p40
      %A41 = getelementptr inbounds double, ptr %A, i32 %p41
      %A42 = getelementptr inbounds double, ptr %A, i32 %p42
      %A43 = getelementptr inbounds double, ptr %A, i32 %p43
      %A44 = getelementptr inbounds double, ptr %A, i32 %p44
      %A45 = getelementptr inbounds double, ptr %A, i32 %p45
      %A46 = getelementptr inbounds double, ptr %A, i32 %p46
      %A47 = getelementptr inbounds double, ptr %A, i32 %p47
      %A48 = getelementptr inbounds double, ptr %A, i32 %p48
      %A49 = getelementptr inbounds double, ptr %A, i32 %p49
      %A50 = getelementptr inbounds double, ptr %A, i32 %p50
      %A51 = getelementptr inbounds double, ptr %A, i32 %p51
      %A52 = getelementptr inbounds double, ptr %A, i32 %p52
      %A53 = getelementptr inbounds double, ptr %A, i32 %p53
      %A54 = getelementptr inbounds double, ptr %A, i32 %p54
      %A55 = getelementptr inbounds double, ptr %A, i32 %p55
      %A56 = getelementptr inbounds double, ptr %A, i32 %p56
      %A57 = getelementptr inbounds double, ptr %A, i32 %p57
      %A58 = getelementptr inbounds double, ptr %A, i32 %p58
      %A59 = getelementptr inbounds double, ptr %A, i32 %p59

      %val0 = load double, ptr %A0
      store double %val0, ptr %A1

      %val2 = load double, ptr %A2
      store double %val2, ptr %A3

      %val4 = load double, ptr %A4
      store double %val4, ptr %A5

      %val6 = load double, ptr %A6
      store double %val6, ptr %A7

      %val8 = load double, ptr %A8
      store double %val8, ptr %A9

      %val10 = load double, ptr %A10
      store double %val10, ptr %A11

      %val12 = load double, ptr %A12
      store double %val12, ptr %A13

      %val13 = load double, ptr %A13
      store double %val13, ptr %A15

      %val16 = load double, ptr %A16
      store double %val16, ptr %A17

      %val18 = load double, ptr %A18
      store double %val18, ptr %A19

      %val20 = load double, ptr %A20
      store double %val20, ptr %A21

      %val22 = load double, ptr %A22
      store double %val22, ptr %A23

      %val24 = load double, ptr %A24
      store double %val24, ptr %A25

      %val26 = load double, ptr %A26
      store double %val26, ptr %A27

      %val28 = load double, ptr %A28
      store double %val28, ptr %A29

      %val30 = load double, ptr %A30
      store double %val30, ptr %A31

      %val32 = load double, ptr %A32
      store double %val32, ptr %A33

      %val34 = load double, ptr %A34
      store double %val34, ptr %A35

      %val36 = load double, ptr %A36
      store double %val36, ptr %A37

      %val38 = load double, ptr %A38
      store double %val38, ptr %A39

      %val40 = load double, ptr %A40
      store double %val40, ptr %A41

      %val42 = load double, ptr %A42
      store double %val42, ptr %A43

      %val44 = load double, ptr %A44
      store double %val44, ptr %A45

      %val46 = load double, ptr %A46
      store double %val46, ptr %A47

      %val48 = load double, ptr %A48
      store double %val48, ptr %A49

      %val50 = load double, ptr %A50
      store double %val50, ptr %A51

      %val52 = load double, ptr %A52
      store double %val52, ptr %A53

      %val54 = load double, ptr %A54
      store double %val54, ptr %A55

      %val56 = load double, ptr %A56
      store double %val56, ptr %A57

      %val58 = load double, ptr %A58
      store double %val58, ptr %A59

      br label %inc


inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: SCoP could not be simplified
