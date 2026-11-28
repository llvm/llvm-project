include "../Commands/OptionsBase.td"

// Options of the option groups that are shared between multiple commands.

let Command = "architecture" in {
  def architecture_arch : Option<"arch", "a">,
                          Group<1>,
                          Arg<"Architecture">,
                          Desc<"Specify the architecture for the target.">;
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
