macro(add_obs_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "OBS executables")
endmacro(add_obs_executable)

