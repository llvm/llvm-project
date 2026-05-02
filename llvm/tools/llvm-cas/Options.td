include "llvm/Option/OptParser.td"

class F<string name> : Flag<["--", "-"], name>;

def grp_action : OptionGroup<"Actions">, HelpText<"llvm-cas actions">;

def help : F<"help">, HelpText<"Prints this help output">;
def : Flag<["-"], "h">, Alias<help>, HelpText<"Alias for --help">;

// Tool actions

def cas_dump : F<"dump">, HelpText<"Dump internal contents">, Group<grp_action>;
def cat_node_data : F<"cat-node-data">,
                    HelpText<"Cat node data">,
                    Group<grp_action>;
def make_blob : F<"make-blob">, HelpText<"Make blob">, Group<grp_action>;
def make_node : F<"make-node">, HelpText<"Make node">, Group<grp_action>;
def ls_node_refs : F<"ls-node-refs">,
                   HelpText<"List node refs">,
                   Group<grp_action>;
def import : F<"import">,
             HelpText<"Import objects from another CAS">,
             Group<grp_action>;
def put_cache_key : F<"put-cache-key">,
                    HelpText<"Set a value for a cache key">,
                    Group<grp_action>;
def get_cache_result : F<"get-cache-result">,
                       HelpText<"Get the result value from a cache key">,
                       Group<grp_action>;
def validate : F<"validate">,
               HelpText<"Validate ObjectStore">,
               Group<grp_action>;
def validate_object : F<"validate-object">,
                      HelpText<"Validate the object for CASID">,
                      Group<grp_action>;
def validate_if_needed : F<"validate-if-needed">,
                         HelpText<"Validate cas contents if needed">,
                         Group<grp_action>;
def prune : F<"prune">, HelpText<"Prune local cas storage">, Group<grp_action>;

// Tool options

def cas_path : Separate<["-", "--"], "cas">,
               MetaVarName<"<path>">,
               HelpText<"Path to CAS on disk">;

def upstream_cas : Separate<["-", "--"], "upstream-cas">,
                   MetaVarName<"<path>">,
                   HelpText<"Path to another upstream CAS">;

def data : Separate<["-", "--"], "data">,
           MetaVarName<"<path>">,
           HelpText<"Path to data or '-' for stdin">;

def check_hash : F<"check-hash">,
                 HelpText<"Check all hashes during validation">;

def allow_recovery : F<"allow-recovery">,
                     HelpText<"Allow recovery of CAS data">;

def force : F<"force">, HelpText<"Force validation even if unnecessary">;

def in_process : F<"in-process">, HelpText<"Validation in-process">;
