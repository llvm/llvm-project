
macro(serialize_lit_param output_var param value)
  string(APPEND ${output_var} "config.${param} = ${value}\n")
endmacro()

macro(serialize_lit_string_param output_var param value)
  # Ensure that all quotes in the value are escaped for a valid python string.
  string(REPLACE "\"" "\\\"" _escaped_value "${value}")
  string(APPEND ${output_var} "config.${param} = \"${_escaped_value}\"\n")
endmacro()

macro(serialize_lit_params_list output_var list)
  foreach(param IN LISTS ${list})
    string(FIND "${param}" "=" _eq_index)
    string(SUBSTRING "${param}" 0 ${_eq_index} name)
    string(SUBSTRING "${param}" ${_eq_index} -1 value)
    string(SUBSTRING "${value}" 1 -1 value) # strip the leading =
    serialize_lit_string_param("${output_var}" "${name}" "${value}")
  endforeach()
endmacro()
