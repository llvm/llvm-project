include "../Commands/OptionsBase.td"

// Options of the option groups that are shared between multiple commands.
//
// Note that the options are emitted sorted by def name, not in the order in
// which they are declared. Some option groups depend on this order, which is
// documented (and checked) at the use sites.

let Command = "architecture" in {
  def architecture_arch : Option<"arch", "a">,
                          Group<1>,
                          Arg<"Architecture">,
                          Desc<"Specify the architecture for the target.">;
}

// OptionGroupFormat::GetDefinitions() returns a slice of these options and
// relies on them being sorted as: count, format, gdb-format, size.
let Command = "format" in {
  def format_count : Option<"count", "c">,
                     Group<4>,
                     Arg<"Count">,
                     Desc<"The number of total items to display.">;
  def format_format : Option<"format", "f">,
                      Group<1>,
                      Arg<"Format">,
                      Desc<"Specify a format to be used for display.">;
  def format_gdb_format
      : Option<"gdb-format", "G">,
        Group<2>,
        Arg<"GDBFormat">,
        Desc<"Specify a format using a GDB format specifier string.">;
  def format_size : Option<"size", "s">,
                    Group<3>,
                    Arg<"ByteSize">,
                    Desc<"The size in bytes to use when displaying with the "
                         "selected format.">;
}

// OptionGroupMemoryTag picks one of the two variants below.
let Command = "memory tag" in {
  def memory_tag_show_tags : Option<"show-tags", "\\x02">,
                             Group<1>,
                             Desc<"Include memory tags in output.">;
}

let Command = "memory tag binary" in {
  def memory_tag_binary_show_tags
      : Option<"show-tags", "\\x02">,
        Group<1>,
        Desc<"Include memory tags in output (does not apply to binary output).">;
}

let Command = "output file" in {
  def output_file_outfile : Option<"outfile", "o">,
                            Group<1>,
                            Arg<"Filename">,
                            Desc<"Specify a path for capturing command output.">;
  def output_file_append_outfile
      : Option<"append-outfile", "\\x01">,
        Group<1>,
        Desc<"Append to the file specified with '--outfile <path>'.">;
}

// OptionGroupPlatform::GetDefinitions() omits the --platform option by
// dropping the first element, so its def name must sort first.
let Command = "platform" in {
  def platform : Option<"platform", "p">,
                 Arg<"Platform">,
                 Desc<"Specify name of the platform to use for this target, "
                      "creating the platform if necessary.">;
  def platform_build : Option<"build", "b">,
                       Arg<"None">,
                       Desc<"Specify the initial SDK build number.">;
  def platform_sysroot : Option<"sysroot", "S">,
                         Arg<"Filename">,
                         Desc<"Specify the SDK root directory that contains a "
                              "root of all remote system files.">;
  def platform_version
      : Option<"version", "v">,
        Arg<"None">,
        Desc<"Specify the initial SDK version to use prior to connecting.">;
}

let Command = "uuid" in {
  def uuid_uuid : Option<"uuid", "u">,
                  Group<1>,
                  Arg<"ModuleUUID">,
                  Desc<"A module UUID value.">;
}

let Command = "value object display" in {
  def value_object_display_depth
      : Option<"depth", "D">,
        Group<1>,
        Arg<"Count">,
        Desc<"Set the max recurse depth when dumping aggregate types (default "
             "is infinity).">;
  def value_object_display_dynamic_type
      : Option<"dynamic-type", "d">,
        Group<1>,
        EnumArg<"DynamicValueType">,
        Desc<"Show the object as its full dynamic type, not its static type, "
             "if available.">;
  def value_object_display_element_count
      : Option<"element-count", "Z">,
        Group<1>,
        Arg<"Count">,
        Desc<"Treat the result of the expression as if its type is an array "
             "of this many values.">;
  def value_object_display_flat
      : Option<"flat", "F">,
        Group<1>,
        Desc<"Display results in a flat format that uses expression paths for "
             "each variable or member.">;
  def value_object_display_location
      : Option<"location", "L">,
        Group<1>,
        Desc<"Show variable location information.">;
  def value_object_display_no_summary_depth
      : Option<"no-summary-depth", "Y">,
        Group<1>,
        OptionalArg<"Count">,
        Desc<"Set the depth at which omitting summary information stops "
             "(default is 1).">;
  def value_object_display_object_description
      : Option<"object-description", "O">,
        Group<1>,
        Desc<"Display using a language-specific description API, if "
             "possible.">;
  def value_object_display_ptr_depth
      : Option<"ptr-depth", "P">,
        Group<1>,
        Arg<"Count">,
        Desc<"The number of pointers to be traversed when dumping values "
             "(default is zero).">;
  def value_object_display_raw_output
      : Option<"raw-output", "R">,
        Group<1>,
        Desc<"Don't use formatting options.">;
  def value_object_display_show_all_children
      : Option<"show-all-children", "A">,
        Group<1>,
        Desc<"Ignore the upper bound on the number of children to show.">;
  def value_object_display_show_types
      : Option<"show-types", "T">,
        Group<1>,
        Desc<"Show variable types when dumping values.">;
  def value_object_display_synthetic_type
      : Option<"synthetic-type", "S">,
        Group<1>,
        Arg<"Boolean">,
        Desc<"Show the object obeying its synthetic provider, if available.">;
  def value_object_display_validate
      : Option<"validate", "V">,
        Group<1>,
        Arg<"Boolean">,
        Desc<"Show results of type validators.">;
}

// OptionGroupVariable::GetDefinitions() omits the frame-only options by
// dropping the leading elements, so their def names must sort first.
let Command = "variable" in {
  def variable_frame_no_args : Option<"no-args", "a">,
                               Groups<[1, 2]>,
                               Desc<"Omit function arguments.">;
  def variable_frame_no_locals : Option<"no-locals", "l">,
                                 Groups<[1, 2]>,
                                 Desc<"Omit local variables.">;
  def variable_frame_no_recognized_args
      : Option<"no-recognized-args", "t">,
        Groups<[1, 2]>,
        Desc<"Omit recognized function arguments.">;
  def variable_frame_show_globals
      : Option<"show-globals", "g">,
        Groups<[1, 2]>,
        Desc<"Show the current frame source file global and static "
             "variables.">;
  // Use 'e' for synthEtic - s and y are both taken.
  def variable_no_synthetic : Option<"no-synthetic", "e">,
                              Groups<[1, 2]>,
                              Desc<"Omit synthetic variables.">;
  def variable_regex : Option<"regex", "r">,
                       Groups<[1, 2]>,
                       Desc<"The <variable-name> argument for name lookups are "
                            "regular expressions.">;
  def variable_scope
      : Option<"scope", "s">,
        Groups<[1, 2]>,
        Desc<"Show variable scope (argument, local, global, static).">;
  def variable_show_declaration
      : Option<"show-declaration", "c">,
        Groups<[1, 2]>,
        Desc<"Show variable declaration information (source file and line "
             "where the variable was declared).">;
  def variable_summary
      : Option<"summary", "y">,
        Group<1>,
        Arg<"Name">,
        Desc<"Specify the summary that the variable output should use.">;
  def variable_summary_string
      : Option<"summary-string", "z">,
        Group<2>,
        Arg<"Name">,
        Desc<"Specify a summary string to use to format the variable output.">;
}

let Command = "watchpoint" in {
  def watchpoint_language : Option<"language", "l">,
                            Group<2>,
                            Arg<"Language">,
                            Desc<"Language of expression to run">;
  def watchpoint_size : Option<"size", "s">,
                        Group<1>,
                        Arg<"ByteSize">,
                        Desc<"Number of bytes to use to watch a region.">;
  def watchpoint_watch : Option<"watch", "w">,
                         Group<1>,
                         EnumArg<"WatchType">,
                         Desc<"Specify the type of watching to perform.">;
}
