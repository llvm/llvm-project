// RUN: %llvmgcc %s -S -o - 

struct tree_common {};

struct tree_int_cst {
 struct tree_common common;
  struct tree_int_cst_lowhi {
    unsigned long long low;
    long long high;
  } int_cst;
};

enum XXX { yyy };

struct tree_function_decl {
  struct tree_common common;
  long long locus, y;
  __extension__ enum  XXX built_in_class : 2;

};


union tree_node {
  struct tree_int_cst int_cst;
  struct tree_function_decl function_decl;
};


void foo (union tree_node * decl) {
  decl->function_decl.built_in_class != 0;
}


