add_lldb_library(lldbPluginObjectContainerBigArchive PLUGIN                     
  ObjectContainerBigArchive.cpp                                                    
                                                                                   
  LINK_LIBS                                                                     
    lldbCore                                                                    
    lldbHost                                                                    
    lldbSymbol                                                                  
  LINK_COMPONENTS                                                               
    Support                                                                     
  )   
