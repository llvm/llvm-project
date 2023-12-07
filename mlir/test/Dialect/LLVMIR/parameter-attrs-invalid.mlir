// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Argument attributes

// expected-error@below {{"llvm.noalias" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_noalias_arg_type(%0 : i32 {llvm.noalias})

// -----

// expected-error@below {{"llvm.noalias" should be a unit attribute}}
llvm.func @invalid_noalias_attr_type(%0 : !llvm.ptr {llvm.noalias = 10 : i32})

// -----

// expected-error@below {{"llvm.readonly" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_readonly_arg_type(%0 : i32 {llvm.readonly})

// -----

// expected-error@below {{"llvm.readonly" should be a unit attribute}}
llvm.func @invalid_readonly_attr_type(%0 : i32 {llvm.readonly = i32})

// -----

// expected-error@below {{"llvm.nest" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_nest_arg_type(%0 : i32 {llvm.nest})

// -----

// expected-error@below {{"llvm.nest" should be a unit attribute}}
llvm.func @invalid_nest_attr_type(%0 : i32 {llvm.nest = "foo"})

// -----

// expected-error@below {{"llvm.align" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_align_arg_type(%0 : i32 {llvm.align = 10 : i32})

// -----

// expected-error@below {{"llvm.align" should be an integer attribute}}
llvm.func @invalid_align_attr_type(%0 : i32 {llvm.align = "foo"})

// -----

// expected-error@below {{"llvm.sret" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_sret_arg_type(%0 : i32 {llvm.sret = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.byval" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_byval_arg_type(%0 : i32 {llvm.byval = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.byval" attribute attached to LLVM pointer argument of different type}}
llvm.func @invalid_byval_attr_type(%0 : !llvm.ptr<!llvm.struct<(f32)>> {llvm.byval = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.byref" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_byref_arg_type(%0 : i32 {llvm.byref = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.byref" attribute attached to LLVM pointer argument of different type}}
llvm.func @invalid_byref_attr_type(%0 : !llvm.ptr<!llvm.struct<(f32)>> {llvm.byref = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.inalloca" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_inalloca_arg_type(%0 : i32 {llvm.inalloca = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.inalloca" attribute attached to LLVM pointer argument of different type}}
llvm.func @invalid_inalloca_attr_type(%0 : !llvm.ptr<!llvm.struct<(f32)>> {llvm.inalloca = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.signext" attribute attached to non-integer LLVM type}}
llvm.func @invalid_signext_arg_type(%0 : f32 {llvm.signext})

// -----

// expected-error@below {{"llvm.signext" should be a unit attribute}}
llvm.func @invalid_signext_attr_type(%0 : i32 {llvm.signext = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.zeroext" attribute attached to non-integer LLVM type}}
llvm.func @invalid_zeroext_arg_type(%0 : f32 {llvm.zeroext})

// -----

// expected-error@below {{"llvm.zeroext" should be a unit attribute}}
llvm.func @invalid_zeroext_attr_type(%0 : i32 {llvm.zeroext = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.noundef" should be a unit attribute}}
llvm.func @invalid_noundef_attr_type(%0 : i32 {llvm.noundef = !llvm.ptr})

// -----

// expected-error@below {{"llvm.dereferenceable" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_dereferenceable_arg_type(%0 : f32 {llvm.dereferenceable = 12 : i64})

// -----

// expected-error@below {{"llvm.dereferenceable" should be an integer attribute}}
llvm.func @invalid_dereferenceable_attr_type(%0 : !llvm.ptr {llvm.dereferenceable = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.dereferenceable_or_null" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_dereferenceable_or_null_arg_type(%0 : f32 {llvm.dereferenceable_or_null = 12 : i64})

// -----

// expected-error@below {{"llvm.dereferenceable_or_null" should be an integer attribute}}
llvm.func @invalid_dereferenceable_or_null_attr_type(%0 : !llvm.ptr {llvm.dereferenceable_or_null = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.inreg" should be a unit attribute}}
llvm.func @invalid_inreg_attr_type(%0 : i32 {llvm.inreg = !llvm.ptr})

// -----

// expected-error@below {{"llvm.nocapture" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_nocapture_arg_type(%0 : f32 {llvm.nocapture})

// -----

// expected-error@below {{"llvm.nocapture" should be a unit attribute}}
llvm.func @invalid_nocapture_attr_type(%0 : !llvm.ptr {llvm.nocapture = f32})

// -----

// expected-error@below {{"llvm.nofree" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_nofree_arg_type(%0 : f32 {llvm.nofree})

// -----

// expected-error@below {{"llvm.nofree" should be a unit attribute}}
llvm.func @invalid_nofree_attr_type(%0 : !llvm.ptr {llvm.nofree = f32})

// -----

// expected-error@below {{"llvm.nonnull" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_nonnull_arg_type(%0 : f32 {llvm.nonnull})

// -----

// expected-error@below {{"llvm.nonnull" should be a unit attribute}}
llvm.func @invalid_nonnull_attr_type(%0 : !llvm.ptr {llvm.nonnull = f32})

// -----

// expected-error@below {{"llvm.preallocated" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_preallocated_arg_type(%0 : f32 {llvm.preallocated = i64})

// -----

// expected-error@below {{"llvm.preallocated" should be a type attribute}}
llvm.func @invalid_preallocated_attr_type(%0 : !llvm.ptr {llvm.preallocated})

// -----

// expected-error@below {{"llvm.returned" should be a unit attribute}}
llvm.func @invalid_returned_attr_type(%0 : i32 {llvm.returned = !llvm.ptr})

// -----

// expected-error@below {{"llvm.alignstack" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_alignstack_arg_type(%0 : i32 {llvm.alignstack = 10 : i32})

// -----

// expected-error@below {{"llvm.alignstack" should be an integer attribute}}
llvm.func @invalid_alignstack_attr_type(%0 : i32 {llvm.alignstack = "foo"})

// -----

// expected-error@below {{"llvm.writeonly" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_writeonly_arg_type(%0 : i32 {llvm.writeonly})

// -----

// expected-error@below {{"llvm.writeonly" should be a unit attribute}}
llvm.func @invalid_writeonly_attr_type(%0 : i32 {llvm.writeonly = i32})

// -----


// Result attributes

// expected-error@below {{expects result attribute array to have the same number of elements as the number of function results, got 1, but expected 0}}
llvm.func @void_def() -> (!llvm.void {llvm.noundef})

// -----

// expected-error @below{{"llvm.align" should be an integer attribute}}
llvm.func @alignattr_ret() -> (!llvm.ptr {llvm.align = 1.0 : f32})

// -----

// expected-error @below{{"llvm.align" attribute attached to non-pointer LLVM type}}
llvm.func @alignattr_ret() -> (i32 {llvm.align = 4})

// -----

// expected-error @below{{"llvm.noalias" should be a unit attribute}}
llvm.func @noaliasattr_ret() -> (!llvm.ptr {llvm.noalias = 1})

// -----

// expected-error @below{{"llvm.noalias" attribute attached to non-pointer LLVM type}}
llvm.func @noaliasattr_ret() -> (i32 {llvm.noalias})

// -----

// expected-error @below{{"llvm.noundef" should be a unit attribute}}
llvm.func @noundefattr_ret() -> (!llvm.ptr {llvm.noundef = 1})

// -----

// expected-error @below{{"llvm.signext" should be a unit attribute}}
llvm.func @signextattr_ret() -> (i32 {llvm.signext = 1})

// -----

// expected-error @below{{"llvm.signext" attribute attached to non-integer LLVM type}}
llvm.func @signextattr_ret() -> (f32 {llvm.signext})

// -----

// expected-error @below{{"llvm.zeroext" should be a unit attribute}}
llvm.func @zeroextattr_ret() -> (i32 {llvm.zeroext = 1})

// -----

// expected-error @below{{"llvm.zeroext" attribute attached to non-integer LLVM type}}
llvm.func @zeroextattr_ret() -> (f32 {llvm.zeroext})

// -----

// expected-error @below{{"llvm.allocalign" is not a valid result attribute}}
llvm.func @allocalign_ret() -> (f32 {llvm.allocalign})

// -----

// expected-error @below{{"llvm.allocptr" is not a valid result attribute}}
llvm.func @allocptr_ret() -> (!llvm.ptr {llvm.allocptr})

// -----

// expected-error @below{{"llvm.byval" is not a valid result attribute}}
llvm.func @byval_ret() -> (!llvm.ptr {llvm.byval = i64})

// -----

// expected-error @below{{"llvm.byref" is not a valid result attribute}}
llvm.func @byref_ret() -> (!llvm.ptr {llvm.byref = i64})

// -----

// expected-error @below{{"llvm.inalloca" is not a valid result attribute}}
llvm.func @inalloca_ret() -> (!llvm.ptr {llvm.inalloca = i64})

// -----

// expected-error @below{{"llvm.nest" is not a valid result attribute}}
llvm.func @nest_ret() -> (!llvm.ptr {llvm.nest})

// -----

// expected-error @below{{"llvm.nocapture" is not a valid result attribute}}
llvm.func @nocapture_ret() -> (!llvm.ptr {llvm.nocapture})

// -----

// expected-error @below{{"llvm.nofree" is not a valid result attribute}}
llvm.func @nofree_ret() -> (!llvm.ptr {llvm.nofree})

// -----

// expected-error @below{{"llvm.preallocated" is not a valid result attribute}}
llvm.func @preallocated_ret() -> (!llvm.ptr {llvm.preallocated = i64})

// -----

// expected-error @below{{"llvm.readnone" is not a valid result attribute}}
llvm.func @readnone_ret() -> (!llvm.ptr {llvm.readnone})

// -----

// expected-error @below{{"llvm.readonly" is not a valid result attribute}}
llvm.func @readonly_ret() -> (!llvm.ptr {llvm.readonly})

// -----

// expected-error @below{{"llvm.alignstack" is not a valid result attribute}}
llvm.func @alignstack_ret() -> (!llvm.ptr {llvm.alignstack = 16 : i64})

// -----

// expected-error @below{{"llvm.sret" is not a valid result attribute}}
llvm.func @sret_ret() -> (!llvm.ptr {llvm.sret = i64})

// -----

// expected-error @below{{"llvm.writeonly" is not a valid result attribute}}
llvm.func @writeonly_ret() -> (!llvm.ptr {llvm.writeonly})
