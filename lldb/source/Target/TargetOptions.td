include "../Commands/OptionsBase.td"

let Command = "platform rsync" in {
  def platform_rsync_ignore_remote_hostname
      : Option<"ignore-remote-hostname", "i">,
        Desc<"Do not automatically fill in the remote hostname when composing "
             "the rsync command.">;
  def platform_rsync_rsync : Option<"rsync", "r">, Desc<"Enable rsync.">;
  def platform_rsync_rsync_opts
      : Option<"rsync-opts", "R">,
        Arg<"CommandName">,
        Desc<"Platform-specific options required for rsync to work.">;
  def platform_rsync_rsync_prefix
      : Option<"rsync-prefix", "P">,
        Arg<"CommandName">,
        Desc<"Platform-specific rsync prefix put before the remote path.">;
}
