// RUN: mlir-opt -split-input-file -verify-diagnostics %s

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_0, base_type = @tbaa_root_0, offset = 0 : i64}
  }
  llvm.func @tbaa(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i8) : i8
    // expected-error@below {{expected '@tbaa_tag_1' to specify a fully qualified reference}}
    llvm.store %0, %arg0 {tbaa = [@tbaa_tag_1]} : i8, !llvm.ptr
    llvm.return
  }
}

// -----

llvm.func @tbaa(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(1 : i8) : i8
  // expected-error@below {{attribute 'tbaa' failed to satisfy constraint: symbol ref array attribute}}
  llvm.store %0, %arg0 {tbaa = ["sym"]} : i8, !llvm.ptr
  llvm.return
}

// -----

module {
  llvm.func @tbaa(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i8) : i8
    // expected-error@below {{expected '@metadata::@domain' to resolve to a llvm.tbaa_tag}}
    llvm.store %0, %arg0 {tbaa = [@metadata::@domain]} : i8, !llvm.ptr
    llvm.return
  }
  llvm.metadata @metadata {
    llvm.alias_scope_domain @domain
  }
}

// -----

module {
  llvm.func @tbaa(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i8) : i8
    // expected-error@below {{expected '@metadata::@sym' to be a valid reference}}
    llvm.store %0, %arg0 {tbaa = [@metadata::@sym]} : i8, !llvm.ptr
    llvm.return
  }
  llvm.metadata @metadata {
  }
}

// -----

llvm.func @tbaa(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(1 : i8) : i8
  // expected-error@below {{expected '@tbaa::@sym' to reference a metadata op}}
  llvm.store %0, %arg0 {tbaa = [@tbaa::@sym]} : i8, !llvm.ptr
  llvm.return
}

// -----

llvm.func @tbaa() {
  // expected-error@below {{expects parent op 'llvm.metadata'}}
  llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
  llvm.return
}

// -----

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
  }

  llvm.func @tbaa() {
    // expected-error@below {{expects parent op 'llvm.metadata'}}
    llvm.tbaa_type_desc @tbaa_type_desc_1 {id = "omnipotent char", members = {<@tbaa_root_0, 0>}}
    llvm.return
  }
}

// -----

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
  }

  llvm.func @tbaa() {
    // expected-error@below {{expects parent op 'llvm.metadata'}}
    llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_0, base_type = @tbaa_root_0, offset = 0 : i64}
    llvm.return
  }
}

// -----

module {
  llvm.metadata @__tbaa {
    // expected-error@below {{expected non-empty "identity"}}
    llvm.tbaa_root @tbaa_root_0 {id = ""}
  }
}

// -----

  "builtin.module"() ({
    "llvm.metadata"() ({
      "llvm.tbaa_root"() {identity = "Simple C/C++ TBAA", sym_name = "tbaa_root_0"} : () -> ()
      "llvm.tbaa_type_desc"() {identity = "omnipotent char", members = [@tbaa_root_0], offsets = array<i64: 0>, sym_name = "tbaa_type_desc_1"} : () -> ()
    // expected-error@below {{expected the same number of elements in "members" and "offsets": 2 != 1}}
      "llvm.tbaa_type_desc"() {identity = "agg_t", members = [@tbaa_type_desc_1, @tbaa_type_desc_1], offsets = array<i64: 0>, sym_name = "tbaa_type_desc_2"} : () -> ()
    }) {sym_name = "__tbaa"} : () -> ()
  }) : () -> ()

// -----

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    // expected-error@below {{expected "base_type" to reference a symbol from 'llvm.metadata @__tbaa' defined by either 'llvm.tbaa_root' or 'llvm.tbaa_type_desc' while it references '@tbaa_root_2'}}
    llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_0, base_type = @tbaa_root_2, offset = 0 : i64}
  }
}

// -----

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    // expected-error@below {{expected "access_type" to reference a symbol from 'llvm.metadata @__tbaa' defined by either 'llvm.tbaa_root' or 'llvm.tbaa_type_desc' while it references '@tbaa_root_2'}}
    llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_2, base_type = @tbaa_root_0, offset = 0 : i64}
  }
}

// -----

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    llvm.tbaa_type_desc @tbaa_type_desc_1 {id = "omnipotent char", members = {<@tbaa_root_0, 0>}}
    llvm.tbaa_type_desc @tbaa_type_desc_2 {id = "long long", members = {<@tbaa_type_desc_1, 0>}}
    // expected-error@below {{expected "members" to reference a symbol from 'llvm.metadata @__tbaa' defined by either 'llvm.tbaa_root' or 'llvm.tbaa_type_desc' while it references '@tbaa_type_desc_4'}}
    llvm.tbaa_type_desc @tbaa_type_desc_3 {id = "agg2_t", members = {<@tbaa_type_desc_2, 0>, <@tbaa_type_desc_4, 8>}}
  }
}

// -----

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_0, base_type = @tbaa_root_0, offset = 0 : i64}
    // expected-error@below {{expected "access_type" to reference a symbol from 'llvm.metadata @__tbaa' defined by either 'llvm.tbaa_root' or 'llvm.tbaa_type_desc' while it references '@tbaa_tag_1'}}
    llvm.tbaa_tag @tbaa_tag_2 {access_type = @tbaa_tag_1, base_type = @tbaa_root_0, offset = 0 : i64}
  }
}

// -----

module {
  // expected-error@below {{has cycle in TBAA graph (graph closure: <tbaa_type_desc_2, tbaa_type_desc_1>)}}
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    llvm.tbaa_type_desc @tbaa_type_desc_1 {id = "omnipotent char", members = {<@tbaa_type_desc_2, 0>}}
    llvm.tbaa_type_desc @tbaa_type_desc_2 {id = "long long", members = {<@tbaa_type_desc_1, 0>}}
  }
}
