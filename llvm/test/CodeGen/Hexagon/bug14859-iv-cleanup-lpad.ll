; RUN: llc -march=hexagon -O3 < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { i32 (...)**, i32, %s.1 }
%s.1 = type { %s.2, %s.5*, %s.6*, i32 }
%s.2 = type { i32 (...)**, i32, i8, i8, i16, i32, i32, %s.3*, %s.4*, i32* }
%s.3 = type { %s.3*, i32, i32, i8* }
%s.4 = type { %s.4*, i32, void (i8, %s.2*, i32)* }
%s.5 = type { i32 (...)**, i8, i32*, i32*, i32**, i32**, i32*, i32*, i32**, i32**, i32*, i32*, i32**, i32**, i32* }
%s.6 = type { i32 (...)**, %s.1 }
%s.7 = type { %s.8, i8 }
%s.8 = type { %s.0* }

define %s.0* @f0(%s.0* %a0, i32* nocapture %a1, i32 %a2, i32 signext %a3) align 2 personality i8* bitcast (i32 (...)* @f11 to i8*) {
b0:
  %v0 = alloca %s.7, align 4
  %v1 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 1
  store i32 0, i32* %v1, align 4, !tbaa !0
  call void @f2(%s.7* %v0, %s.0* %a0, i1 zeroext true)
  %v2 = getelementptr inbounds %s.7, %s.7* %v0, i32 0, i32 1
  %v3 = load i8, i8* %v2, align 4, !tbaa !4, !range !6
  %v4 = icmp ne i8 %v3, 0
  %v5 = icmp sgt i32 %a2, 0
  %v6 = and i1 %v4, %v5
  %v7 = bitcast %s.0* %a0 to i8**
  br i1 %v6, label %b2, label %b1

b1:                                               ; preds = %b0
  %v8 = bitcast %s.0* %a0 to i8*
  br label %b16

b2:                                               ; preds = %b0
  %v9 = load i8*, i8** %v7, align 4, !tbaa !7
  %v10 = getelementptr i8, i8* %v9, i32 -12
  %v11 = bitcast i8* %v10 to i32*
  %v12 = load i32, i32* %v11, align 4
  %v13 = bitcast %s.0* %a0 to i8*
  %v14 = add i32 %v12, 32
  %v15 = getelementptr inbounds i8, i8* %v13, i32 %v14
  %v16 = bitcast i8* %v15 to %s.5**
  %v17 = load %s.5*, %s.5** %v16, align 4, !tbaa !9
  %v18 = invoke signext i32 @f3(%s.5* %v17)
          to label %b3 unwind label %b7

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b13, %b3
  %v19 = phi i32 [ %v68, %b13 ], [ %v18, %b3 ]
  %v20 = phi i32 [ %v55, %b13 ], [ %a2, %b3 ]
  %v21 = phi i32* [ %v59, %b13 ], [ %a1, %b3 ]
  %v22 = icmp eq i32 %v19, -1
  br i1 %v22, label %b15, label %b10

b5:                                               ; preds = %b16, %b9
  %v23 = landingpad { i8*, i32 }
          cleanup
  %v24 = extractvalue { i8*, i32 } %v23, 0
  %v25 = extractvalue { i8*, i32 } %v23, 1
  br label %b18

b6:                                               ; preds = %b13
  %v26 = landingpad { i8*, i32 }
          catch i8* null
  br label %b8

b7:                                               ; preds = %b11, %b2
  %v27 = phi i32* [ %v21, %b11 ], [ %a1, %b2 ]
  %v28 = landingpad { i8*, i32 }
          catch i8* null
  br label %b8

b8:                                               ; preds = %b7, %b6
  %v29 = phi i32* [ %v59, %b6 ], [ %v27, %b7 ]
  %v30 = phi { i8*, i32 } [ %v26, %b6 ], [ %v28, %b7 ]
  %v31 = extractvalue { i8*, i32 } %v30, 0
  %v32 = call i8* @f9(i8* %v31) #0
  %v33 = load i8*, i8** %v7, align 4, !tbaa !7
  %v34 = getelementptr i8, i8* %v33, i32 -12
  %v35 = bitcast i8* %v34 to i32*
  %v36 = load i32, i32* %v35, align 4
  %v37 = getelementptr inbounds i8, i8* %v13, i32 %v36
  %v38 = bitcast i8* %v37 to %s.1*
  %v39 = add i32 %v36, 8
  %v40 = getelementptr inbounds i8, i8* %v13, i32 %v39
  %v41 = load i8, i8* %v40, align 1, !tbaa !11
  %v42 = or i8 %v41, 4
  invoke void @f6(%s.1* %v38, i8 zeroext %v42, i1 zeroext true)
          to label %b9 unwind label %b14

b9:                                               ; preds = %b8
  invoke void @f10()
          to label %b16 unwind label %b5

b10:                                              ; preds = %b4
  %v43 = icmp eq i32 %v19, %a3
  br i1 %v43, label %b11, label %b12

b11:                                              ; preds = %b10
  %v44 = load i32, i32* %v1, align 4, !tbaa !0
  %v45 = add nsw i32 %v44, 1
  store i32 %v45, i32* %v1, align 4, !tbaa !0
  %v46 = load i8*, i8** %v7, align 4, !tbaa !7
  %v47 = getelementptr i8, i8* %v46, i32 -12
  %v48 = bitcast i8* %v47 to i32*
  %v49 = load i32, i32* %v48, align 4
  %v50 = add i32 %v49, 32
  %v51 = getelementptr inbounds i8, i8* %v13, i32 %v50
  %v52 = bitcast i8* %v51 to %s.5**
  %v53 = load %s.5*, %s.5** %v52, align 4, !tbaa !9
  %v54 = invoke signext i32 @f4(%s.5* %v53)
          to label %b16 unwind label %b7

b12:                                              ; preds = %b10
  %v55 = add nsw i32 %v20, -1
  %v56 = icmp slt i32 %v55, 1
  br i1 %v56, label %b15, label %b13

b13:                                              ; preds = %b12
  %v57 = load i32, i32* %v1, align 4, !tbaa !0
  %v58 = add nsw i32 %v57, 1
  store i32 %v58, i32* %v1, align 4, !tbaa !0
  %v59 = getelementptr inbounds i32, i32* %v21, i32 1
  store i32 %v19, i32* %v21, align 4, !tbaa !13
  %v60 = load i8*, i8** %v7, align 4, !tbaa !7
  %v61 = getelementptr i8, i8* %v60, i32 -12
  %v62 = bitcast i8* %v61 to i32*
  %v63 = load i32, i32* %v62, align 4
  %v64 = add i32 %v63, 32
  %v65 = getelementptr inbounds i8, i8* %v13, i32 %v64
  %v66 = bitcast i8* %v65 to %s.5**
  %v67 = load %s.5*, %s.5** %v66, align 4, !tbaa !9
  %v68 = invoke signext i32 @f5(%s.5* %v67)
          to label %b4 unwind label %b6

b14:                                              ; preds = %b8
  %v69 = landingpad { i8*, i32 }
          cleanup
  %v70 = extractvalue { i8*, i32 } %v69, 0
  %v71 = extractvalue { i8*, i32 } %v69, 1
  invoke void @f10()
          to label %b18 unwind label %b20

b15:                                              ; preds = %b12, %b4
  %v72 = phi i8 [ 2, %b12 ], [ 1, %b4 ]
  br label %b16

b16:                                              ; preds = %b15, %b11, %b9, %b1
  %v73 = phi i8* [ %v8, %b1 ], [ %v13, %b11 ], [ %v13, %b9 ], [ %v13, %b15 ]
  %v74 = phi i8 [ 0, %b1 ], [ 0, %b11 ], [ 0, %b9 ], [ %v72, %b15 ]
  %v75 = phi i32* [ %a1, %b1 ], [ %v21, %b11 ], [ %v29, %b9 ], [ %v21, %b15 ]
  store i32 0, i32* %v75, align 4, !tbaa !13
  %v76 = load i8*, i8** %v7, align 4, !tbaa !7
  %v77 = getelementptr i8, i8* %v76, i32 -12
  %v78 = bitcast i8* %v77 to i32*
  %v79 = load i32, i32* %v78, align 4
  %v80 = getelementptr inbounds i8, i8* %v73, i32 %v79
  %v81 = bitcast i8* %v80 to %s.1*
  %v82 = load i32, i32* %v1, align 4, !tbaa !0
  %v83 = icmp eq i32 %v82, 0
  %v84 = or i8 %v74, 2
  %v85 = select i1 %v83, i8 %v84, i8 %v74
  invoke void @f7(%s.1* %v81, i8 zeroext %v85, i1 zeroext false)
          to label %b17 unwind label %b5

b17:                                              ; preds = %b16
  %v86 = getelementptr inbounds %s.7, %s.7* %v0, i32 0, i32 0
  call void @f1(%s.8* %v86)
  ret %s.0* %a0

b18:                                              ; preds = %b14, %b5
  %v87 = phi i8* [ %v24, %b5 ], [ %v70, %b14 ]
  %v88 = phi i32 [ %v25, %b5 ], [ %v71, %b14 ]
  %v89 = getelementptr inbounds %s.7, %s.7* %v0, i32 0, i32 0
  invoke void @f1(%s.8* %v89)
          to label %b19 unwind label %b20

b19:                                              ; preds = %b18
  %v90 = insertvalue { i8*, i32 } undef, i8* %v87, 0
  %v91 = insertvalue { i8*, i32 } %v90, i32 %v88, 1
  resume { i8*, i32 } %v91

b20:                                              ; preds = %b18, %b14
  %v92 = landingpad { i8*, i32 }
          catch i8* null
  call void @f8() #1
  unreachable
}

declare void @f1(%s.8* nocapture) unnamed_addr align 2

declare void @f2(%s.7* nocapture, %s.0*, i1 zeroext) unnamed_addr align 2

declare signext i32 @f3(%s.5*) align 2

declare signext i32 @f4(%s.5*) align 2

declare signext i32 @f5(%s.5*) align 2

declare void @f6(%s.1*, i8 zeroext, i1 zeroext) align 2

declare void @f7(%s.1*, i8 zeroext, i1 zeroext) align 2

declare void @f8()

declare i8* @f9(i8*)

declare void @f10()

declare i32 @f11(...)

attributes #0 = { nounwind }
attributes #1 = { noreturn nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"bool", !2}
!6 = !{i8 0, i8 2}
!7 = !{!8, !8, i64 0}
!8 = !{!"vtable pointer", !3}
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !2}
!11 = !{!12, !12, i64 0}
!12 = !{!"_ZTSNSt5_IosbIiE8_IostateE", !2}
!13 = !{!14, !14, i64 0}
!14 = !{!"wchar_t", !2}
