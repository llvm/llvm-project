## Show that when bitcode is added to an archive it is handled correctly.
## The symbol table is as expected and it can be extracted without issue.

# RUN: rm -rf %t
# RUN: split-file %s %t && mkdir -p %t/extracted
# RUN: cd %t
# RUN: llvm-as a.ll -o a.bc

## Create symtab from bitcode for a new archive.
# RUN: llvm-ar rcs new.a a.bc
# RUN: llvm-ar t new.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap new.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

# FILES: a.bc

# SYMS:      Archive map
# SYMS-NEXT: gfunc in a.bc
# SYMS-NEXT: gdata in a.bc

# SYMS:      a.bc:
# SYMS-NEXT: -------- D gdata
# SYMS-NEXT: -------- T gfunc
# SYMS-NEXT: -------- d ldata
# SYMS-NEXT: -------- t lfunc

## Update symtab from bitcode in an existing archive.
# RUN: llvm-ar rcS update.a a.bc
# RUN: llvm-ar t update.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
## Check no symbol table is present.
# RUN: llvm-nm --print-armap update.a | FileCheck --check-prefix=NOSYMS %s --implicit-check-not={{.}}
# RUN: llvm-ar s update.a
# RUN: llvm-ar t update.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap update.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

# NOSYMS: a.bc:
# NOSYMS-NEXT: -------- D gdata
# NOSYMS-NEXT: -------- T gfunc
# NOSYMS-NEXT: -------- d ldata
# NOSYMS-NEXT: -------- t lfunc

## Create symtab from bitcode for a regular archive via MRI script.
# RUN: llvm-ar -M < add.mri
# RUN: llvm-ar t mri.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap mri.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

## Create symtab from bitcode for a new thin archive.
# RUN: llvm-ar rcs --thin new-thin.a a.bc
# RUN: llvm-ar t new-thin.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap new-thin.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

## Update symtab from bitcode in an existing thin archive.
# RUN: llvm-ar rcS --thin update-thin.a a.bc
# RUN: llvm-ar t update-thin.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
## Check no symbol table is present.
# RUN: llvm-nm --print-armap update-thin.a | FileCheck --check-prefix=NOSYMS %s --implicit-check-not={{.}}
# RUN: llvm-ar s update-thin.a
# RUN: llvm-ar t update-thin.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap update-thin.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

## Create symtab from bitcode for a thin archive via MRI script.
# RUN: llvm-ar -M < add-thin.mri
# RUN: llvm-ar t mri-thin.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap mri-thin.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

## Create symtab from bitcode from another archive.
# RUN: llvm-ar rcs input.a a.bc
# RUN: llvm-ar cqsL lib.a input.a
# RUN: llvm-ar t lib.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap lib.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

## Create symtab from bitcode from another archive via MRI script.
# RUN: llvm-ar -M < addlib.mri
# RUN: llvm-ar t mri-addlib.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap mri-addlib.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

## Create symtab from bitcode from another thin archive.
# RUN: llvm-ar rcs --thin input-thin.a a.bc
# RUN: llvm-ar cqsL --thin lib-thin.a input-thin.a
# RUN: llvm-ar t lib-thin.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap lib-thin.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

## Create symtab from bitcode from another thin archive via MRI script.
# RUN: llvm-ar -M < addlib-thin.mri
# RUN: llvm-ar t mri-addlib-thin.a | FileCheck --check-prefix=FILES %s --implicit-check-not={{.}}
# RUN: llvm-nm --print-armap mri-addlib-thin.a | FileCheck --check-prefix=SYMS %s --implicit-check-not={{.}}

## Extract bitcode and ensure it has not been changed.
# RUN: cd extracted
# RUN: llvm-ar x ../new.a a.bc
# RUN: cmp a.bc a.bc

#--- a.ll
@gdata = global i32 0
@ldata = internal global i32 0
define void @gfunc() { ret void }
define internal void @lfunc() { ret void }

#--- add.mri
CREATE mri.a
ADDMOD a.bc
SAVE
END

#--- add-thin.mri
CREATETHIN mri-thin.a
ADDMOD a.bc
SAVE
END

#--- addlib.mri
CREATE mri-addlib.a
ADDLIB input.a
SAVE
END

#--- addlib-thin.mri
CREATE mri-addlib-thin.a
ADDLIB input-thin.a
SAVE
END
