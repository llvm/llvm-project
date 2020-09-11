// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// This file has tests for range lists and range pieces. Some use the
// deprecated '-' range punctuation just to be sure it still works.

// These are tests for bits ranges.

def bit_range_hyphen {
  bits<16> field1;
  let field1{15, 14, 13, 12} = {1, 0, 1, 0};
  let field1{11-8} = {1, 0, 1, 1};
  let field1{+7-4} = {1, 1, 0, 0};
  let field1{+3-+0} = {1, 1, 0, 1};
  bit hyphen_field1_ok = !eq(field1, 0xABCD);
}

def bit_range_dotdotdot {
  bits<16> field1;
  let field1{15, 14, 13, 12} = {1, 0, 1, 0};
  let field1{11...8} = {1, 0, 1, 1};
  let field1{+7...4} = {1, 1, 0, 0};
  let field1{+3...+0} = {1, 1, 0, 1};
  bit dotdotdot_field1_ok = !eq(field1, 0xABCD);
}

if !eq(bit_range_hyphen.field1, bit_range_dotdotdot.field1) then
  def bit_range_ok {}
else
  def bit_range_not_ok {}

// These are tests for lists.

def list_range_hyphen {
  list<string> field1 = ["foo", "bar", "baz", "snork", "quux", "quuux",
                         "bazola", "ztesch", "bletch", "flarp"];
  list<string> subfielda = field1[0, 1, 2, 3];
  list<string> subfieldb = field1[4-5];
  list<string> subfieldc = field1[+6-7];
  list<string> subfieldd = field1[+8-+9];
  bit hyphen_subfields_ok = !and(!eq(subfieldb[0], "quux"),
                                 !eq(subfieldd[1], "flarp"));
}

def list_range_dotdotdot {
  list<string> field1 = ["foo", "bar", "baz", "snork", "quux", "quuux",
                         "bazola", "ztesch", "bletch", "flarp"];
  list<string> subfielda = field1[0, 1, 2, 3];
  list<string> subfieldb = field1[4...5];
  list<string> subfieldc = field1[+6...7];
  list<string> subfieldd = field1[+8...+9];
  bit dotdotdot_subfields_ok = !and(!eq(subfieldb[0], "quux"),
                                    !eq(subfieldd[1], "flarp"));
}

if !eq(!head(list_range_hyphen.subfieldd),
       !head(list_range_dotdotdot.subfieldd)) then
  def list_range_ok {}
else
  def list_range_not_ok {}

// This is a test of foreach.

foreach i = {0-3} in 
  foreach j = {4...5} in
    def eachrec#i#j {
      int fi = i;
      int fj = j;
    }

//CHECK: bit dotdotdot_field1_ok = 1
//CHECK: bit hyphen_field1_ok = 1
//CHECK: def bit_range_ok {

//CHECK: def eachrec04 {
//CHECK: def eachrec35 {

//CHECK: bit dotdotdot_subfields_ok = 1
//CHECK: bit hyphen_subfields_ok = 1
//CHECK: def list_range_ok {

