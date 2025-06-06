#  Copyright 2006 John Maddock, Paul A. Bristow and Xiaogang Zhang.
#  Distributed under the Boost Software License, Version 1.0.
#  (See accompanying file LICENSE_1_0.txt or copy at
#  http://www.boost.org/LICENSE_1_0.txt).
#
# Example makefile that builds the docs.
# Note that all the following paths will have to be changed to match
# your actual installation paths.
#

# Path to quickbook executable:
QB="C:/download/open/xml/bin/quickbook.exe"

# Path to xsltproc:
XSLTPROC="C:/download/open/xml/bin/xsltproc-win32/xsltproc.exe"

# Path to Boost Trunc:
BOOST=c:/data/boost/boost/trunk

# Path to FO processor (XEP):
FO=C:/Progra~1/xep/xep.bat

# Configuration options:
COMMON_XSL_PARAM=--stringparam admon.graphics "1" --stringparam body.start.indent "0pt" --stringparam chunk.first.sections "1" --stringparam chunk.section.depth "10" --stringparam fop.extensions "0" --stringparam generate.section.toc.level "10" --stringparam html.stylesheet "../../../../../../trunk/doc/html/boostbook.css" --stringparam navig.graphics "1" --stringparam page.margin.inner "0.5in" --stringparam page.margin.outer "0.5in" --stringparam paper.type "A4" --stringparam toc.max.depth "4" --stringparam toc.section.depth "10" --stringparam xep.extensions "1"
PDF_XSL_PARAM=--stringparam admon.graphics.extension ".svg" --stringparam use.role.for.mediaobject 1 --stringparam preferred.mediaobject.role print --stringparam admon.graphics.path "../html/images/"
HTML_XSL_PARAM=
PROJECT_NAME=math

all : pdf html

pdf : pdf/$(PROJECT_NAME).pdf
html : html/index.html

xml/$(PROJECT_NAME).xml :
	-mkdir xml
	$(QB) --output-file=xml\$(PROJECT_NAME).xml $(PROJECT_NAME).qbk

xml/$(PROJECT_NAME).docbook : xml\$(PROJECT_NAME).xml xml/catalog.xml
	set XML_CATALOG_FILES=xml/catalog.xml
	$(XSLTPROC) $(COMMON_XSL_PARAM) --xinclude -o "xml\$(PROJECT_NAME).docbook" "$(BOOST)\tools\boostbook\xsl\docbook.xsl" "xml\$(PROJECT_NAME).xml"

xml/$(PROJECT_NAME).fo : xml\$(PROJECT_NAME).docbook xml/catalog.xml
	 set XML_CATALOG_FILES=xml/catalog.xml
	 $(XSLTPROC) $(COMMON_XSL_PARAM) $(PDF_XSL_PARAM) --xinclude -o "xml\$(PROJECT_NAME).fo" "$(BOOST)\tools\boostbook\xsl\fo.xsl" "xml\$(PROJECT_NAME).docbook"
  
pdf/$(PROJECT_NAME).pdf : xml\$(PROJECT_NAME).fo
	-mkdir pdf
	set JAVA_HOME=C:/PROGRA~1/Java/j2re1.4.2_12
	call $(FO) xml\$(PROJECT_NAME).fo pdf\$(PROJECT_NAME).pdf

html/index.html : xml\$(PROJECT_NAME).fo
	-mkdir html
	 set XML_CATALOG_FILES=xml/catalog.xml
	 $(XSLTPROC) $(COMMON_XSL_PARAM) $(HTML_XSL_PARAM) --xinclude -o "html/" "$(BOOST)\tools\boostbook\xsl\html.xsl" "xml\$(PROJECT_NAME).docbook"

xml/catalog.xml :
	@echo <<xml/catalog.xml
<?xml version="1.0"?>
<!DOCTYPE catalog
  PUBLIC "-//OASIS/DTD Entity Resolution XML Catalog V1.0//EN"
  "http://www.oasis-open.org/committees/entity/release/1.0/catalog.dtd">
<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
  <rewriteURI uriStartString="http://www.boost.org/tools/boostbook/dtd/" rewritePrefix="file:///$(BOOST)/tools/boostbook/dtd/"/>
  <rewriteURI uriStartString="http://docbook.sourceforge.net/release/xsl/current/" rewritePrefix="file:///C:/download/open/xml/docbook-xsl-snapshot/"/>
  <rewriteURI uriStartString="http://www.oasis-open.org/docbook/xml/4.2/" rewritePrefix="file:///C:/download/open/xml/docbook-xml/"/>
</catalog>
<<






