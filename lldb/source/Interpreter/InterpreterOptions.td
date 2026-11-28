include "../Commands/OptionsBase.td"

// Options of the option groups that are shared between multiple commands.

let Command = "architecture" in {
  def architecture_arch : Option<"arch", "a">,
                          Group<1>,
                          Arg<"Architecture">,
                          Desc<"Specify the architecture for the target.">;
}

let Command = "uuid" in {
  def uuid_uuid : Option<"uuid", "u">,
                  Group<1>,
                  Arg<"ModuleUUID">,
                  Desc<"A module UUID value.">;
}
