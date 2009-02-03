//===- Common.td - Common definitions for LLVMC2  ----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions used in llvmc2 tool description files.
//
//===----------------------------------------------------------------------===//

class Tool<list<dag> l> {
      list<dag> properties = l;
}

// Possible Tool properties.

def in_language;
def out_language;
def output_suffix;
def cmd_line;
def join;
def sink;
def actions;

// Possible option types.

def alias_option;
def switch_option;
def parameter_option;
def parameter_list_option;
def prefix_option;
def prefix_list_option;

// Possible option properties.

def extern;
def help;
def hidden;
def multi_val;
def one_or_more;
def really_hidden;
def required;
def zero_or_one;

// Empty DAG marker.
def empty;

// The 'case' construct.
def case;

// Boolean operators.
def and;
def or;

// Primitive tests.
def switch_on;
def parameter_equals;
def element_in_list;
def input_languages_contain;
def not_empty;
def default;

// Possible actions.

def append_cmd;
def forward;
def forward_as;
def stop_compilation;
def unpack_values;
def error;

// Increase/decrease the edge weight.
def inc_weight;
def dec_weight;

// Used to specify plugin priority.
class PluginPriority<int p> {
      int priority = p;
}

// Option list - used to specify aliases and sometimes help strings.
class OptionList<list<dag> l> {
      list<dag> options = l;
}

// Map from suffixes to language names

class LangToSuffixes<string str, list<string> lst> {
      string lang = str;
      list<string> suffixes = lst;
}

class LanguageMap<list<LangToSuffixes> lst> {
      list<LangToSuffixes> map = lst;
}

// Compilation graph

class EdgeBase<string t1, string t2, dag d> {
      string a = t1;
      string b = t2;
      dag weight = d;
}

class Edge<string t1, string t2> : EdgeBase<t1, t2, (empty)>;

// Edge and SimpleEdge are synonyms.
class SimpleEdge<string t1, string t2> : EdgeBase<t1, t2, (empty)>;

// Optionally enabled edge.
class OptionalEdge<string t1, string t2, dag props> : EdgeBase<t1, t2, props>;

class CompilationGraph<list<EdgeBase> lst> {
      list<EdgeBase> edges = lst;
}
