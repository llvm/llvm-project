include "../Commands/OptionsBase.td"

// Options of the option groups that are shared between multiple commands.

let Command = "architecture" in {
  def architecture_arch : Option<"arch", "a">,
                          Group<1>,
                          Arg<"Architecture">,
                          Desc<"Specify the architecture for the target.">;
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

let Command = "uuid" in {
  def uuid_uuid : Option<"uuid", "u">,
                  Group<1>,
                  Arg<"ModuleUUID">,
                  Desc<"A module UUID value.">;
}
