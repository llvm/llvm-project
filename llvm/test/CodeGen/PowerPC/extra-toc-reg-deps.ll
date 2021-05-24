; RUN: llc -verify-machineinstrs -mcpu=pwr8 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux"

%"class.Foam::messageStream.6" = type <{ %"class.Foam::string.5", i32, i32, i32, [4 x i8] }>
%"class.Foam::string.5" = type { %"class.std::basic_string.4" }
%"class.std::basic_string.4" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider.3" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider.3" = type { i8* }
%"class.Foam::prefixOSstream.27" = type { %"class.Foam::OSstream.26", i8, %"class.Foam::string.5" }
%"class.Foam::OSstream.26" = type { %"class.Foam::Ostream.base.9", %"class.Foam::fileName.10", %"class.std::basic_ostream.25"* }
%"class.Foam::Ostream.base.9" = type <{ %"class.Foam::IOstream.8", i16 }>
%"class.Foam::IOstream.8" = type { i32 (...)**, i32, [4 x i8], %"class.Foam::IOstream::versionNumber.7", i32, i32, i32, i32 }
%"class.Foam::IOstream::versionNumber.7" = type <{ double, i32, [4 x i8] }>
%"class.Foam::fileName.10" = type { %"class.Foam::string.5" }
%"class.std::basic_ostream.25" = type { i32 (...)**, %"class.std::basic_ios.24" }
%"class.std::basic_ios.24" = type { %"class.std::ios_base.16", %"class.std::basic_ostream.25"*, i8, i8, %"class.std::basic_streambuf.17"*, %"class.std::ctype.21"*, %"class.std::__gnu_cxx_ldbl128::num_put.22"*, %"class.std::__gnu_cxx_ldbl128::num_get.23"* }
%"class.std::ios_base.16" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list.11"*, %"struct.std::ios_base::_Words.12", [8 x %"struct.std::ios_base::_Words.12"], i32, %"struct.std::ios_base::_Words.12"*, %"class.std::locale.15" }
%"struct.std::ios_base::_Callback_list.11" = type { %"struct.std::ios_base::_Callback_list.11"*, void (i32, %"class.std::ios_base.16"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words.12" = type { i8*, i64 }
%"class.std::locale.15" = type { %"class.std::locale::_Impl.14"* }
%"class.std::locale::_Impl.14" = type { i32, %"class.std::locale::facet.13"**, i64, %"class.std::locale::facet.13"**, i8** }
%"class.std::locale::facet.13" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf.17" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale.15" }
%"class.std::ctype.21" = type <{ %"class.std::locale::facet.base.18", [4 x i8], %struct.__locale_struct.20*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base.18" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct.20 = type { [13 x %struct.__locale_data.19*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data.19 = type opaque
%"class.std::__gnu_cxx_ldbl128::num_put.22" = type { %"class.std::locale::facet.base.18", [4 x i8] }
%"class.std::__gnu_cxx_ldbl128::num_get.23" = type { %"class.std::locale::facet.base.18", [4 x i8] }
%"class.Foam::primitiveMesh.135" = type { i32 (...)**, i32, i32, i32, i32, i32, i32, i32, i32, i32, %"class.Foam::List.116"*, %"class.Foam::List.0"*, %"class.Foam::List.1"*, %"class.Foam::List.1"*, %"class.Foam::List.1"*, %"class.Foam::List.5"*, %"class.Foam::List.1"*, %"class.Foam::List.1"*, %"class.Foam::List.1"*, %"class.Foam::List.1"*, %"class.Foam::List.1"*, %"class.Foam::List.1"*, %"class.Foam::List.1"*, %"class.Foam::DynamicList.40", %"class.Foam::HashSet.127", %"class.Foam::Field.131"*, %"class.Foam::Field.131"*, %"class.Foam::Field.11"*, %"class.Foam::Field.131"* }
%"class.Foam::List.116" = type opaque
%"class.Foam::List.0" = type { %"class.Foam::UList.119" }
%"class.Foam::UList.119" = type { i32, %"class.Foam::edge.118"* }
%"class.Foam::edge.118" = type { %"class.Foam::FixedList.117" }
%"class.Foam::FixedList.117" = type { [2 x i32] }
%"class.Foam::List.5" = type { %"class.Foam::UList.6" }
%"class.Foam::UList.6" = type { i32, %"class.Foam::cell.121"* }
%"class.Foam::cell.121" = type { %"class.Foam::List.3" }
%"class.Foam::List.3" = type { %"class.Foam::UList.4" }
%"class.Foam::UList.4" = type { i32, i32* }
%"class.Foam::List.1" = type { %"class.Foam::UList.2" }
%"class.Foam::UList.2" = type { i32, %"class.Foam::List.3"* }
%"class.Foam::DynamicList.40" = type <{ %"class.Foam::List.3", i32, [4 x i8] }>
%"class.Foam::HashSet.127" = type { %"class.Foam::HashTable.7" }
%"class.Foam::HashTable.7" = type { i32, i32, %"struct.Foam::HashTable<Foam::nil, int, Foam::Hash<Foam::label> >::hashedEntry.125"** }
%"struct.Foam::HashTable<Foam::nil, int, Foam::Hash<Foam::label> >::hashedEntry.125" = type <{ i32, [4 x i8], %"struct.Foam::HashTable<Foam::nil, int, Foam::Hash<Foam::label> >::hashedEntry.125"*, %"class.Foam::nil.124", [7 x i8] }>
%"class.Foam::nil.124" = type { i8 }
%"class.Foam::Field.11" = type { %"class.Foam::refCount.128", %"class.Foam::List.12" }
%"class.Foam::refCount.128" = type { i32 }
%"class.Foam::List.12" = type { %"class.Foam::UList.13" }
%"class.Foam::UList.13" = type { i32, double* }
%"class.Foam::Field.131" = type { %"class.Foam::refCount.128", %"class.Foam::List.8" }
%"class.Foam::List.8" = type { %"class.Foam::UList.9" }
%"class.Foam::UList.9" = type { i32, %"class.Foam::Vector.29"* }
%"class.Foam::Vector.29" = type { %"class.Foam::VectorSpace.10" }
%"class.Foam::VectorSpace.10" = type { [3 x double] }
%"class.Foam::Ostream.189" = type <{ %"class.Foam::IOstream.8", i16, [6 x i8] }>

@_ZN4Foam4InfoE = external global %"class.Foam::messageStream.6", align 8
@.str27 = external unnamed_addr constant [24 x i8], align 1
@.str28 = external unnamed_addr constant [7 x i8], align 1
@_ZN4Foam4PoutE = external global %"class.Foam::prefixOSstream.27", align 8

define void @_ZN4Foam13checkTopologyERKNS_8polyMeshEbb(i1 zeroext %allTopology) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br i1 undef, label %for.body, label %for.cond.cleanup

; CHECK-LABEL: @_ZN4Foam13checkTopologyERKNS_8polyMeshEbb

; CHECK: addis [[REG1:[0-9]+]], 2, .LC0@toc@ha
; CHECK: std 2, 40(1)
; CHECK: ld {{[0-9]+}}, .LC0@toc@l([[REG1]])
; CHECK: {{mr|ld}} 2,
; CHECK: mtctr
; CHECK: bctrl
; CHECK: ld 2, 40(1)

; CHECK: std 2, 40(1)
; CHECK: {{mr|ld}} 2,
; CHECK: mtctr
; CHECK: bctrl
; CHECK: ld 2, 40(1)

for.cond.cleanup:                                 ; preds = %entry
  br i1 undef, label %if.then.i, label %if.else.i

if.then.i:                                        ; preds = %for.cond.cleanup
  br i1 undef, label %if.then.i1435, label %if.else.i1436

if.else.i:                                        ; preds = %for.cond.cleanup
  unreachable

if.then.i1435:                                    ; preds = %if.then.i
  br label %_ZN4Foam12returnReduceIiNS_5sumOpIiEEEET_RKS3_RKT0_ii.exit

if.else.i1436:                                    ; preds = %if.then.i
  br label %_ZN4Foam12returnReduceIiNS_5sumOpIiEEEET_RKS3_RKT0_ii.exit

_ZN4Foam12returnReduceIiNS_5sumOpIiEEEET_RKS3_RKT0_ii.exit: ; preds = %if.else.i1436, %if.then.i1435
  br i1 undef, label %for.body.i, label %_ZNK4Foam8ZoneMeshINS_8cellZoneENS_8polyMeshEE15checkDefinitionEb.exit

for.body:                                         ; preds = %entry
  unreachable

for.body.i:                                       ; preds = %_ZN4Foam12returnReduceIiNS_5sumOpIiEEEET_RKS3_RKT0_ii.exit
  unreachable

_ZNK4Foam8ZoneMeshINS_8cellZoneENS_8polyMeshEE15checkDefinitionEb.exit: ; preds = %_ZN4Foam12returnReduceIiNS_5sumOpIiEEEET_RKS3_RKT0_ii.exit
  br i1 undef, label %for.body.i1480, label %_ZNK4Foam8ZoneMeshINS_8faceZoneENS_8polyMeshEE15checkDefinitionEb.exit

for.body.i1480:                                   ; preds = %_ZNK4Foam8ZoneMeshINS_8cellZoneENS_8polyMeshEE15checkDefinitionEb.exit
  unreachable

_ZNK4Foam8ZoneMeshINS_8faceZoneENS_8polyMeshEE15checkDefinitionEb.exit: ; preds = %_ZNK4Foam8ZoneMeshINS_8cellZoneENS_8polyMeshEE15checkDefinitionEb.exit
  br i1 undef, label %for.body.i1504, label %_ZNK4Foam8ZoneMeshINS_9pointZoneENS_8polyMeshEE15checkDefinitionEb.exit

for.body.i1504:                                   ; preds = %_ZNK4Foam8ZoneMeshINS_8faceZoneENS_8polyMeshEE15checkDefinitionEb.exit
  unreachable

_ZNK4Foam8ZoneMeshINS_9pointZoneENS_8polyMeshEE15checkDefinitionEb.exit: ; preds = %_ZNK4Foam8ZoneMeshINS_8faceZoneENS_8polyMeshEE15checkDefinitionEb.exit
  invoke void @_ZN4Foam4word12stripInvalidEv()
          to label %_ZN4Foam4wordC2EPKcb.exit unwind label %lpad.i

lpad.i:                                           ; preds = %_ZNK4Foam8ZoneMeshINS_9pointZoneENS_8polyMeshEE15checkDefinitionEb.exit
  %0 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %0

_ZN4Foam4wordC2EPKcb.exit:                        ; preds = %_ZNK4Foam8ZoneMeshINS_9pointZoneENS_8polyMeshEE15checkDefinitionEb.exit
  invoke void @_ZN4Foam7cellSetC1ERKNS_8polyMeshERKNS_4wordEiNS_8IOobject11writeOptionE()
          to label %invoke.cont59 unwind label %lpad

invoke.cont59:                                    ; preds = %_ZN4Foam4wordC2EPKcb.exit
  br i1 undef, label %_ZNSsD2Ev.exit, label %if.then.i.i, !prof !1

if.then.i.i:                                      ; preds = %invoke.cont59
  br i1 true, label %if.then.i.i.i1508, label %if.else.i.i.i

if.then.i.i.i1508:                                ; preds = %if.then.i.i
  br label %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i

if.else.i.i.i:                                    ; preds = %if.then.i.i
  br label %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i

_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i: ; preds = %if.else.i.i.i, %if.then.i.i.i1508
  br i1 undef, label %if.then4.i.i, label %_ZNSsD2Ev.exit

if.then4.i.i:                                     ; preds = %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i
  br label %_ZNSsD2Ev.exit

_ZNSsD2Ev.exit:                                   ; preds = %if.then4.i.i, %_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii.exit.i.i, %invoke.cont59
  br i1 undef, label %for.body70, label %for.cond.cleanup69

for.cond.cleanup69:                               ; preds = %_ZNSsD2Ev.exit
  br i1 undef, label %if.then121, label %if.else

lpad:                                             ; preds = %_ZN4Foam4wordC2EPKcb.exit
  %1 = landingpad { i8*, i32 }
          cleanup
  br i1 undef, label %_ZNSsD2Ev.exit1578, label %if.then.i.i1570, !prof !1

if.then.i.i1570:                                  ; preds = %lpad
  br i1 undef, label %if.then4.i.i1577, label %_ZNSsD2Ev.exit1578

if.then4.i.i1577:                                 ; preds = %if.then.i.i1570
  unreachable

_ZNSsD2Ev.exit1578:                               ; preds = %if.then.i.i1570, %lpad
  unreachable

for.body70:                                       ; preds = %_ZNSsD2Ev.exit
  unreachable

if.then121:                                       ; preds = %for.cond.cleanup69
  unreachable

if.else:                                          ; preds = %for.cond.cleanup69
  invoke void @_ZN4Foam4word12stripInvalidEv()
          to label %_ZN4Foam4wordC2EPKcb.exit1701 unwind label %lpad.i1689

lpad.i1689:                                       ; preds = %if.else
  %2 = landingpad { i8*, i32 }
          cleanup
  unreachable

_ZN4Foam4wordC2EPKcb.exit1701:                    ; preds = %if.else
  invoke void @_ZN4Foam8pointSetC1ERKNS_8polyMeshERKNS_4wordEiNS_8IOobject11writeOptionE()
          to label %invoke.cont169 unwind label %lpad165

invoke.cont169:                                   ; preds = %_ZN4Foam4wordC2EPKcb.exit1701
  %call177 = invoke zeroext i1 undef(%"class.Foam::primitiveMesh.135"* undef, i1 zeroext true, %"class.Foam::HashSet.127"* undef)
          to label %invoke.cont176 unwind label %lpad175

invoke.cont176:                                   ; preds = %invoke.cont169
  br i1 %call177, label %if.then178, label %if.end213

if.then178:                                       ; preds = %invoke.cont176
  unreachable

lpad165:                                          ; preds = %_ZN4Foam4wordC2EPKcb.exit1701
  %3 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad175:                                          ; preds = %invoke.cont169
  %4 = landingpad { i8*, i32 }
          cleanup
  invoke void @_ZN4Foam8pointSetD1Ev()
          to label %eh.resume unwind label %terminate.lpad

if.end213:                                        ; preds = %invoke.cont176
  invoke void @_ZN4Foam4word12stripInvalidEv()
          to label %_ZN4Foam4wordC2EPKcb.exit1777 unwind label %lpad.i1765

lpad.i1765:                                       ; preds = %if.end213
  %5 = landingpad { i8*, i32 }
          cleanup
  br i1 undef, label %eh.resume.i1776, label %if.then.i.i.i1767, !prof !1

if.then.i.i.i1767:                                ; preds = %lpad.i1765
  unreachable

eh.resume.i1776:                                  ; preds = %lpad.i1765
  resume { i8*, i32 } %5

_ZN4Foam4wordC2EPKcb.exit1777:                    ; preds = %if.end213
  invoke void @_ZN4Foam7faceSetC1ERKNS_8polyMeshERKNS_4wordEiNS_8IOobject11writeOptionE()
          to label %invoke.cont221 unwind label %lpad217

invoke.cont221:                                   ; preds = %_ZN4Foam4wordC2EPKcb.exit1777
  br i1 undef, label %_ZNSsD2Ev.exit1792, label %if.then.i.i1784, !prof !1

if.then.i.i1784:                                  ; preds = %invoke.cont221
  br i1 undef, label %if.then4.i.i1791, label %_ZNSsD2Ev.exit1792

if.then4.i.i1791:                                 ; preds = %if.then.i.i1784
  br label %_ZNSsD2Ev.exit1792

_ZNSsD2Ev.exit1792:                               ; preds = %if.then4.i.i1791, %if.then.i.i1784, %invoke.cont221
  %call232 = invoke zeroext i1 undef(%"class.Foam::primitiveMesh.135"* undef, i1 zeroext true, %"class.Foam::HashSet.127"* undef)
          to label %invoke.cont231 unwind label %lpad230

invoke.cont231:                                   ; preds = %_ZNSsD2Ev.exit1792
  invoke void @_ZN4Foam6reduceIiNS_5sumOpIiEEEEvRKNS_4ListINS_8UPstream11commsStructEEERT_RKT0_ii()
          to label %invoke.cont243 unwind label %lpad230

lpad217:                                          ; preds = %_ZN4Foam4wordC2EPKcb.exit1777
  %6 = landingpad { i8*, i32 }
          cleanup
  br label %eh.resume

lpad230:                                          ; preds = %invoke.cont231, %_ZNSsD2Ev.exit1792
  %7 = landingpad { i8*, i32 }
          cleanup
  invoke void @_ZN4Foam7faceSetD1Ev()
          to label %eh.resume unwind label %terminate.lpad

invoke.cont243:                                   ; preds = %invoke.cont231
  invoke void @_ZN4Foam4word12stripInvalidEv()
          to label %_ZN4Foam4wordC2EPKcb.exit1862 unwind label %lpad.i1850

lpad.i1850:                                       ; preds = %invoke.cont243
  %8 = landingpad { i8*, i32 }
          cleanup
  unreachable

_ZN4Foam4wordC2EPKcb.exit1862:                    ; preds = %invoke.cont243
  invoke void @_ZN4Foam7faceSetC1ERKNS_8polyMeshERKNS_4wordEiNS_8IOobject11writeOptionE()
          to label %invoke.cont280 unwind label %lpad276

invoke.cont280:                                   ; preds = %_ZN4Foam4wordC2EPKcb.exit1862
  br i1 undef, label %_ZNSsD2Ev.exit1877, label %if.then.i.i1869, !prof !1

if.then.i.i1869:                                  ; preds = %invoke.cont280
  unreachable

_ZNSsD2Ev.exit1877:                               ; preds = %invoke.cont280
  br i1 undef, label %if.then292, label %if.end328

if.then292:                                       ; preds = %_ZNSsD2Ev.exit1877
  unreachable

lpad276:                                          ; preds = %_ZN4Foam4wordC2EPKcb.exit1862
  %9 = landingpad { i8*, i32 }
          cleanup
  unreachable

if.end328:                                        ; preds = %_ZNSsD2Ev.exit1877
  br i1 %allTopology, label %if.then331, label %if.end660

if.then331:                                       ; preds = %if.end328
  unreachable

if.end660:                                        ; preds = %if.end328
  invoke void @_ZN4Foam13messageStreamcvRNS_8OSstreamEEv()
          to label %invoke.cont668 unwind label %lpad663

invoke.cont668:                                   ; preds = %if.end660
  %call671 = invoke dereferenceable(56) %"class.Foam::Ostream.189"* @_ZN4FoamlsERNS_7OstreamEPKc()
          to label %invoke.cont670 unwind label %lpad663

invoke.cont670:                                   ; preds = %invoke.cont668
  invoke void @_ZN4FoamlsERNS_7OstreamEi()
          to label %invoke.cont674 unwind label %lpad663

invoke.cont674:                                   ; preds = %invoke.cont670
  %call677 = invoke dereferenceable(56) %"class.Foam::Ostream.189"* @_ZN4FoamlsERNS_7OstreamEPKc()
          to label %invoke.cont676 unwind label %lpad663

invoke.cont676:                                   ; preds = %invoke.cont674
  invoke void undef(%"class.Foam::Ostream.189"* %call677)
          to label %if.end878 unwind label %lpad663

lpad663:                                          ; preds = %invoke.cont670, %if.end660, %invoke.cont668, %invoke.cont674, %invoke.cont676
  %10 = landingpad { i8*, i32 }
          cleanup
  br i1 undef, label %_ZN4Foam4ListIiED2Ev.exit.i3073, label %delete.notnull.i.i3071

if.end878:                                        ; preds = %invoke.cont676
  br i1 undef, label %_ZN4Foam11regionSplitD2Ev.exit, label %delete.notnull.i.i3056

delete.notnull.i.i3056:                           ; preds = %if.end878
  unreachable

_ZN4Foam11regionSplitD2Ev.exit:                   ; preds = %if.end878
  br i1 undef, label %if.then883, label %if.else888

if.then883:                                       ; preds = %_ZN4Foam11regionSplitD2Ev.exit
  unreachable

delete.notnull.i.i3071:                           ; preds = %lpad663
  unreachable

_ZN4Foam4ListIiED2Ev.exit.i3073:                  ; preds = %lpad663
  invoke void @_ZN4Foam11regIOobjectD2Ev()
          to label %eh.resume unwind label %terminate.lpad

if.else888:                                       ; preds = %_ZN4Foam11regionSplitD2Ev.exit
  invoke void @_ZN4Foam4word12stripInvalidEv()
          to label %_ZN4Foam4wordC2EPKcb.exit3098 unwind label %lpad.i3086

lpad.i3086:                                       ; preds = %if.else888
  %11 = landingpad { i8*, i32 }
          cleanup
  unreachable

_ZN4Foam4wordC2EPKcb.exit3098:                    ; preds = %if.else888
  invoke void @_ZN4Foam8pointSetC1ERKNS_8polyMeshERKNS_4wordEiNS_8IOobject11writeOptionE()
          to label %invoke.cont902 unwind label %lpad898

invoke.cont902:                                   ; preds = %_ZN4Foam4wordC2EPKcb.exit3098
  br i1 undef, label %_ZNSsD2Ev.exit3113, label %if.then.i.i3105, !prof !1

if.then.i.i3105:                                  ; preds = %invoke.cont902
  br i1 undef, label %if.then4.i.i3112, label %_ZNSsD2Ev.exit3113

if.then4.i.i3112:                                 ; preds = %if.then.i.i3105
  unreachable

_ZNSsD2Ev.exit3113:                               ; preds = %if.then.i.i3105, %invoke.cont902
  %call.i31163117 = invoke zeroext i32 undef(%"class.Foam::IOstream.8"* getelementptr inbounds (%"class.Foam::prefixOSstream.27", %"class.Foam::prefixOSstream.27"* @_ZN4Foam4PoutE, i64 0, i32 0, i32 0, i32 0))
          to label %call.i3116.noexc unwind label %lpad905.loopexit.split-lp

call.i3116.noexc:                                 ; preds = %_ZNSsD2Ev.exit3113
  %call5.i3118 = invoke zeroext i32 null(%"class.Foam::IOstream.8"* getelementptr inbounds (%"class.Foam::prefixOSstream.27", %"class.Foam::prefixOSstream.27"* @_ZN4Foam4PoutE, i64 0, i32 0, i32 0, i32 0), i32 zeroext undef)
          to label %invoke.cont906 unwind label %lpad905.loopexit.split-lp

invoke.cont906:                                   ; preds = %call.i3116.noexc
  unreachable

lpad898:                                          ; preds = %_ZN4Foam4wordC2EPKcb.exit3098
  %12 = landingpad { i8*, i32 }
          cleanup
  br i1 undef, label %_ZNSsD2Ev.exit3204, label %if.then.i.i3196, !prof !1

if.then.i.i3196:                                  ; preds = %lpad898
  unreachable

_ZNSsD2Ev.exit3204:                               ; preds = %lpad898
  unreachable

lpad905.loopexit.split-lp:                        ; preds = %call.i3116.noexc, %_ZNSsD2Ev.exit3113
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  invoke void @_ZN4Foam8pointSetD1Ev()
          to label %eh.resume unwind label %terminate.lpad

eh.resume:                                        ; preds = %_ZN4Foam4ListIiED2Ev.exit.i3073, %lpad230, %lpad175, %lpad905.loopexit.split-lp, %lpad217
  resume { i8*, i32 } undef

terminate.lpad:                                   ; preds = %_ZN4Foam4ListIiED2Ev.exit.i3073, %lpad230, %lpad175, %lpad905.loopexit.split-lp
  %13 = landingpad { i8*, i32 }
          catch i8* null
  unreachable
}

declare dereferenceable(56) %"class.Foam::Ostream.189"* @_ZN4FoamlsERNS_7OstreamEPKc() #0

declare void @_ZN4Foam13messageStreamcvRNS_8OSstreamEEv() #0

declare i32 @__gxx_personality_v0(...)

declare void @_ZN4Foam7cellSetC1ERKNS_8polyMeshERKNS_4wordEiNS_8IOobject11writeOptionE() #0

declare void @_ZN4FoamlsERNS_7OstreamEi() #0

declare void @_ZN4Foam8pointSetC1ERKNS_8polyMeshERKNS_4wordEiNS_8IOobject11writeOptionE() #0

declare void @_ZN4Foam8pointSetD1Ev() #0

declare void @_ZN4Foam7faceSetC1ERKNS_8polyMeshERKNS_4wordEiNS_8IOobject11writeOptionE() #0

declare void @_ZN4Foam7faceSetD1Ev() #0

; Function Attrs: inlinehint
declare void @_ZN4Foam4word12stripInvalidEv() #1 align 2

declare void @_ZN4Foam11regIOobjectD2Ev() #0

declare void @_ZN4Foam6reduceIiNS_5sumOpIiEEEEvRKNS_4ListINS_8UPstream11commsStructEEERT_RKT0_ii() #0

attributes #0 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"branch_weights", i32 64, i32 4}
