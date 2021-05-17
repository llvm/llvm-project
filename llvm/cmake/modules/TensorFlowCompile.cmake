# Ensure the ${model} is available at ${final_path}.
#
function(tfgetmodel model final_path)
  if (IS_ABSOLUTE ${model})
    set(${final_path} ${model} PARENT_SCOPE)
  else()
    set(${final_path}
      ${CMAKE_CURRENT_SOURCE_DIR}/${model} PARENT_SCOPE)
  endif()
endfunction()

# Run the tensorflow compiler (saved_model_cli) on the saved model in the
# ${model} directory, looking for the ${tag_set} tag set, and the SignatureDef
# ${signature_def_key}.
# Produce a pair of files called ${fname}.h and  ${fname}.o in the
# ${CMAKE_CURRENT_BINARY_DIR}. The generated header will define a C++ class
# called ${cpp_class} - which may be a namespace-qualified class name.
function(tfcompile model tag_set signature_def_key fname cpp_class)
  set(prefix ${CMAKE_CURRENT_BINARY_DIR}/${fname})
  set(obj_file ${prefix}.o)
  set(hdr_file ${prefix}.h)
  string(TOUPPER ${fname} fname_allcaps)
  set(override_header ${LLVM_OVERRIDE_MODEL_HEADER_${fname_allcaps}})
  set(override_object ${LLVM_OVERRIDE_MODEL_OBJECT_${fname_allcaps}})
  if (EXISTS "${override_header}" AND EXISTS "${override_object}")
    configure_file(${override_header} ${hdr_file} COPYONLY)
    configure_file(${override_object} ${obj_file} COPYONLY)
    message("Using provided header "
      ${hdr_file} " and object "   ${obj_file}
      " files for model " ${model})
  else()
    tfgetmodel(${model} LLVM_ML_MODELS_ABSOLUTE)
    message("Using model at " ${LLVM_ML_MODELS_ABSOLUTE})
    add_custom_command(OUTPUT ${obj_file} ${hdr_file}
      COMMAND "XLA_FLAGS=\"--xla_cpu_multi_thread_eigen=false\"" ${TENSORFLOW_AOT_COMPILER} aot_compile_cpu
            --dir ${LLVM_ML_MODELS_ABSOLUTE}
            --tag_set ${tag_set}
            --signature_def_key ${signature_def_key}
            --output_prefix ${prefix}
            --cpp_class ${cpp_class}
            --target_triple ${LLVM_HOST_TRIPLE}
    )
  endif()

  # Aggregate the objects so that results of different tfcompile calls may be
  # grouped into one target.
  set(GENERATED_OBJS ${GENERATED_OBJS} ${obj_file} PARENT_SCOPE)
  set_source_files_properties(${obj_file} PROPERTIES
    GENERATED 1 EXTERNAL_OBJECT 1)

  set(GENERATED_HEADERS ${GENERATED_HEADERS} ${hdr_file} PARENT_SCOPE)
  set_source_files_properties(${hdr_file} PROPERTIES
    GENERATED 1)
  
endfunction()
