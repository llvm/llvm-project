; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s

; The Peephole optimizer should fold the load into the cmp even with debug info.
; CHECK-LABEL: _ZN3Foo3batEv
; CHECK-NOT: movq pfoo
; CHECK: cmpq {{%[a-z]+}}, pfoo(%rip)
;
; CHECK-LABEL: _Z3bazv
; CHECK-NOT: movq wibble2
; CHECK: cmpq {{%[a-z]+}}, wibble2(%rip)

; Regenerate test with this command: 
;   clang -emit-llvm -S -O2 -g
; from this source:
;   struct Foo {
;     bool bat();
;     bool operator==(Foo &arg) { return (this == &arg); }
;   };
;   Foo *pfoo;
;   bool Foo::bat() { return (*this == *pfoo); }
;
;   struct Wibble {
;     int x;
;   } *wibble1, *wibble2;
;   struct Flibble {
;     void bar(Wibble *c) {
;       if (c < wibble2)
;         wibble2 = 0;
;       c->x = 0;
;     }
;   } flibble;
;   void baz() { flibble.bar(wibble1); }

%struct.Foo = type { i8 }
%struct.Wibble = type { i32 }
%struct.Flibble = type { i8 }

@pfoo = dso_local global %struct.Foo* null, align 8
@wibble1 = dso_local global %struct.Wibble* null, align 8
@wibble2 = dso_local global %struct.Wibble* null, align 8
@flibble = dso_local global %struct.Flibble zeroinitializer, align 1

; Function Attrs: nounwind readonly uwtable
define zeroext i1 @_ZN3Foo3batEv(%struct.Foo* %this) #0 align 2 {
entry:
  %0 = load %struct.Foo*, %struct.Foo** @pfoo, align 8
  tail call void @llvm.dbg.value(metadata %struct.Foo* %0, i64 0, metadata !62, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  %cmp.i = icmp eq %struct.Foo* %0, %this
  ret i1 %cmp.i
}

; Function Attrs: nounwind uwtable
define dso_local void @_Z3bazv() #1 {
entry:
  %0 = load %struct.Wibble*, %struct.Wibble** @wibble1, align 8
  tail call void @llvm.dbg.value(metadata %struct.Flibble* undef, i64 0, metadata !65, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  %1 = load %struct.Wibble*, %struct.Wibble** @wibble2, align 8
  %cmp.i = icmp ugt %struct.Wibble* %1, %0
  br i1 %cmp.i, label %if.then.i, label %_ZN7Flibble3barEP6Wibble.exit

if.then.i:                                        ; preds = %entry
  store %struct.Wibble* null, %struct.Wibble** @wibble2, align 8
  br label %_ZN7Flibble3barEP6Wibble.exit

_ZN7Flibble3barEP6Wibble.exit:                    ; preds = %entry, %if.then.i
  %x.i = getelementptr inbounds %struct.Wibble, %struct.Wibble* %0, i64 0, i32 0
  store i32 0, i32* %x.i, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind readonly uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!1 = distinct !DISubprogram()

!17 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: null)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: null)
!62 = !DILocalVariable(name: "arg", line: 4, arg: 2, scope: !1, type: !17)
!64 = !{%struct.Flibble* undef}
!65 = !DILocalVariable(name: "this", line: 13, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !1, type: !45)
