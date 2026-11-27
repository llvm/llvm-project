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

let Command = "platform ssh" in {
  def platform_ssh_ssh : Option<"ssh", "s">, Desc<"Enable SSH.">;
  def platform_ssh_ssh_opts
      : Option<"ssh-opts", "S">,
        Arg<"CommandName">,
        Desc<"Platform-specific options required for SSH to work.">;
}

let Command = "platform caching" in {
  def platform_caching_local_cache_dir
      : Option<"local-cache-dir", "c">,
        Arg<"Path">,
        Desc<"Path in which to store local copies of files.">;
}
