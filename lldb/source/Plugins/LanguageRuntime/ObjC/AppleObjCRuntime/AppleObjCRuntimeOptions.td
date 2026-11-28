include "../../../../../source/Commands/OptionsBase.td"

let Command = "language objc class_table dump" in {
  def objc_class_table_dump_verbose
      : Option<"verbose", "v">,
        Desc<"Print ivar and method information in detail">;
}
