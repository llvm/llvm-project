llc test command

```
llc -march=parasol -global-sel -run-pass=instruction-select
```

If you want to see why a pattern isn't imported

```
llvm-tblgen -I llvm/include/ -I llvm/lib/Target/Parasol llvm/lib/Target/Parasol/Parasol.td --gen-global-isel -warn-on-skipped-patterns --stats
```

When you do get to adding tests, they can be written in LLVM IR and then tested with a command like

```
llc -global-isel -march=parasol -stop-after=irtranslator -simplify-mir add.ll
```

where add.ll is

```
define void @add(ptr readonly %a, ptr readonly %b, ptr writeonly %output) local_unnamed_addr #0 {
  %0 = load i8, ptr %a, align 1, !tbaa !3
  %1 = load i8, ptr %b, align 1, !tbaa !3
  %add = add i8 %1, %0
  store i8 %add, ptr %output, align 1, !tbaa !3
  ret void
}
```

# License

Modified by Sunscreen under the AGPLv3 license; see the README at the repository root for more information
