include "llvm/Option/OptParser.td"

// Subcommands

def sc_dump : SubCommand<"dump", "Dump internal contents">;
def sc_cat_node_data : SubCommand<"cat-node-data", "Cat node data">;
def sc_make_blob : SubCommand<"make-blob", "Make blob">;
def sc_make_node : SubCommand<"make-node", "Make node">;
def sc_ls_node_refs : SubCommand<"ls-node-refs", "List node refs">;
def sc_import : SubCommand<"import", "Import objects from another CAS">;
def sc_put_cache_key : SubCommand<"put-cache-key",
                                  "Set a value for a cache key">;
def sc_get_cache_result : SubCommand<"get-cache-result",
                                     "Get the result value from a cache key">;
def sc_validate : SubCommand<"validate", "Validate ObjectStore">;
def sc_validate_object : SubCommand<"validate-object",
                                    "Validate the object for CASID">;
def sc_validate_if_needed : SubCommand<"validate-if-needed",
                                       "Validate cas contents if needed">;
def sc_prune : SubCommand<"prune", "Prune local cas storage">;

defvar sc_all = [sc_dump, sc_cat_node_data, sc_make_blob, sc_make_node,
                 sc_ls_node_refs, sc_import, sc_put_cache_key,
                 sc_get_cache_result, sc_validate, sc_validate_object,
                 sc_validate_if_needed, sc_prune];

class F<string name, list<SubCommand> subcommands = []>
    : Flag<["--", "-"], name, subcommands>;

def help : F<"help">, HelpText<"Prints this help output">;
def : Flag<["-"], "h">, Alias<help>, HelpText<"Alias for --help">;

// Tool options

def cas_path : Separate<["-", "--"], "cas", sc_all>,
               MetaVarName<"<path>">,
               HelpText<"Path to CAS on disk">;

def cas_plugin_path : Separate<["-", "--"], "fcas-plugin-path", sc_all>,
                      MetaVarName<"<path>">,
                      HelpText<"Path to plugin CAS library">;

def cas_plugin_option : Separate<["-", "--"], "fcas-plugin-option", sc_all>,
                        MetaVarName<"<name>=<value>">,
                        HelpText<"Option passed to the plugin CAS">;

def upstream_cas : Separate<["-", "--"], "upstream-cas", [sc_import]>,
                   MetaVarName<"<path>">,
                   HelpText<"Path to another upstream CAS">;

def data : Separate<["-", "--"], "data", [sc_make_blob, sc_make_node]>,
           MetaVarName<"<path>">,
           HelpText<"Path to data or '-' for stdin">;

def check_hash : F<"check-hash", [sc_validate, sc_validate_if_needed]>,
                 HelpText<"Check all hashes during validation">;

def allow_recovery : F<"allow-recovery", [sc_validate_if_needed]>,
                     HelpText<"Allow recovery of CAS data">;

def force : F<"force", [sc_validate_if_needed]>,
            HelpText<"Force validation even if unnecessary">;

def in_process : F<"in-process", [sc_validate_if_needed]>,
                 HelpText<"Validation in-process">;

def verbose : F<"verbose", [sc_validate_if_needed]>,
              HelpText<"Print why validation failed when recovering">;
