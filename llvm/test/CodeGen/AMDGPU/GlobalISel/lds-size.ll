; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa < %S/../lds-size.ll | FileCheck -check-prefix=ALL -check-prefix=HSA %S/../lds-size.ll
